"""Execute retrieval SQL against an isolated, socket-only PostgreSQL cluster."""

import os
import shlex
import shutil
import subprocess
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import orjson
import pytest
import pytest_asyncio
from pgvector import Vector
from psycopg import AsyncConnection, AsyncCursor, Connection
from psycopg.adapt import Dumper
from psycopg.rows import dict_row
from pydantic import JsonValue

from dean_research_tools import Lifecycle, PGTools


class _VectorDumper(Dumper):
    def dump(self, value: Vector) -> bytes:
        return value.to_text().encode()


_SCHEMA = """
CREATE SCHEMA ai_proj_browser;
CREATE SCHEMA ai_proj;
-- A text-backed test vector with cosine distance avoids requiring pgvector
-- server binaries; production SQL and parameter binding still execute unchanged.
CREATE DOMAIN vector AS text;
CREATE FUNCTION cosine_distance(vector, vector) RETURNS double precision
LANGUAGE SQL IMMUTABLE AS $$
    SELECT 1 - sum(x*y) / sqrt(sum(x*x) * sum(y*y))
    FROM unnest(translate($1, '[]', '{}')::double precision[],
                translate($2, '[]', '{}')::double precision[]) v(x, y)
$$;
CREATE OPERATOR <=> (
    LEFTARG = vector, RIGHTARG = vector, FUNCTION = cosine_distance
);
CREATE TABLE ai_proj_browser.doc_content (
    blake3 bytea PRIMARY KEY, doc_type text, saved_path text, metadata jsonb
);
CREATE TABLE ai_proj_browser.tasks (
    task_id bigint PRIMARY KEY, status text, starting_task text
);
CREATE TABLE ai_proj_browser.docs (
    doc_id bigint PRIMARY KEY, task_id bigint REFERENCES ai_proj_browser.tasks,
    blake3 bytea REFERENCES ai_proj_browser.doc_content,
    url text, title text, discovered_context text, downloaded boolean
);
CREATE TABLE ai_proj_browser.links (
    task_id bigint, doc_id bigint REFERENCES ai_proj_browser.docs, status text
);
CREATE VIEW ai_proj_browser.tasks_docs AS
    SELECT task_id, doc_id FROM ai_proj_browser.docs
    UNION
    SELECT task_id, doc_id FROM ai_proj_browser.links WHERE status = 'done';
CREATE TABLE ai_proj_browser.text_content (
    chunk_id bigint PRIMARY KEY,
    blake3 bytea NOT NULL REFERENCES ai_proj_browser.doc_content,
    page integer, chunk_indx integer, text_chunk text, embedding vector
);
CREATE TABLE ai_proj_browser.image_content (
    chunk_id bigint PRIMARY KEY,
    blake3 bytea NOT NULL REFERENCES ai_proj_browser.doc_content
);
CREATE TABLE ai_proj_browser.tasks_done (task_id bigint, history jsonb);
CREATE TABLE ai_proj.research_browser_batches (
    batch_id bigint, research_id bigint, publisher_task_id bigint,
    published_at timestamptz, resumed_at timestamptz
);
CREATE TABLE ai_proj.research_task_pause (batch_id bigint);

INSERT INTO ai_proj_browser.tasks VALUES
    (1, 'complete', 'original'), (7, 'complete', 'duplicates'),
    (8, 'complete', 'other'), (9, 'complete', 'reused'),
    (10, 'complete', 'images');
INSERT INTO ai_proj_browser.doc_content VALUES
    ('\\xaaaa', 'pdf', 'shared.pdf', '{}'),
    ('\\xbbbb', 'html', 'other.html', '{}'),
    ('\\xcccc', 'pdf', 'images.pdf', '{}');
INSERT INTO ai_proj_browser.docs (doc_id, task_id, blake3, url, title) VALUES
    (10, 1, '\\xaaaa', 'https://original.test/a', 'Original'),
    (20, 7, '\\xaaaa', 'https://task.test/a', 'Task source'),
    (30, 7, '\\xaaaa', 'https://alias.test/a', 'Alias source'),
    (40, 8, '\\xbbbb', 'https://other.test/b', 'Other'),
    (50, 9, NULL, 'https://pending.test', 'Not downloaded'),
    (60, 10, '\\xcccc', 'https://images.test', 'Images');
INSERT INTO ai_proj_browser.links VALUES
    (7, 20, 'done'), (7, 20, 'done'), (7, 30, 'done'),
    (9, 30, 'done'), (9, 30, 'done'), (9, 40, 'queued');
INSERT INTO ai_proj_browser.text_content VALUES
    (101, '\\xaaaa', 1, 0, 'query first', '[1,0]'),
    (102, '\\xaaaa', 1, 1, 'query second', '[1,0]'),
    (103, '\\xaaaa', 2, 0, 'query third', '[1,0]'),
    (104, '\\xbbbb', 1, 0, 'query other', '[0,1]');
INSERT INTO ai_proj_browser.image_content VALUES
    (201, '\\xaaaa'), (202, '\\xaaaa'), (203, '\\xbbbb'), (204, '\\xcccc');
"""


@pytest.fixture(scope="module")
def content_database() -> Iterator[str]:
    initdb = shutil.which("initdb")
    if initdb is None:
        candidates = sorted(Path("/usr/lib/postgresql").glob("*/bin/initdb"))
        if len(candidates) > 0:
            initdb = str(candidates[-1])
    if initdb is None or os.geteuid() == 0:
        pytest.skip("isolated PostgreSQL tests require initdb and a non-root user")
    binaries = Path(initdb).parent
    directory = Path(".pytest_cache", f"content-pg-{uuid4().hex[:8]}").resolve()
    directory.mkdir(parents=True)
    data = directory / "data"
    started = False
    try:
        subprocess.run(
            [initdb, "-D", str(data), "-A", "trust", "--no-locale", "-E", "UTF8"],
            check=True, capture_output=True, text=True,
        )
        options = shlex.join([
            "-c", "listen_addresses=",
            "-c", f"unix_socket_directories={directory}",
            "-c", "fsync=off",
        ])
        subprocess.run(
            [str(binaries / "pg_ctl"), "-D", str(data), "-l", str(directory / "log"),
             "-o", options, "-w", "start"],
            check=True, capture_output=True, text=True,
        )
        started = True
        dsn = f"host={directory} dbname=postgres"
        with Connection.connect(dsn, autocommit=True) as conn:
            conn.execute(_SCHEMA)
        yield dsn
    finally:
        if started:
            subprocess.run(
                [str(binaries / "pg_ctl"), "-D", str(data), "-m", "immediate",
                 "-w", "stop"],
                check=True, capture_output=True, text=True,
            )
        shutil.rmtree(directory)


@pytest_asyncio.fixture
async def content_tools(
    content_database: str,
) -> AsyncIterator[tuple[PGTools, Lifecycle]]:
    async with await AsyncConnection.connect(
        content_database, autocommit=True, row_factory=dict_row
    ) as conn:
        conn.adapters.register_dumper(Vector, _VectorDumper)
        tools = object.__new__(PGTools)
        tools.embeddings = SimpleNamespace(
            embed_search=AsyncMock(return_value=[1.0, 0.0])
        )

        @asynccontextmanager
        async def get_cur() -> AsyncIterator[AsyncCursor[dict[str, JsonValue]]]:
            async with conn.cursor() as cur:
                yield cur

        tools._get_cur = get_cur
        yield tools, Lifecycle(conn)


@pytest.mark.asyncio
@pytest.mark.parametrize("semantic", [False, True])
@pytest.mark.parametrize(
    ("task_id", "expected_doc", "expected_task"),
    [(None, 10, 1), (7, 20, 7), (9, 30, 9)],
)
async def test_duplicate_sources_emit_unique_chunks_and_matching_provenance(
    content_tools: tuple[PGTools, Lifecycle],
    semantic: bool,
    task_id: int | None,
    expected_doc: int,
    expected_task: int,
) -> None:
    tools, _ = content_tools
    if semantic:
        result = await tools.semantic_content_search(
            "query", browser_task_id=task_id, top_k=2
        )
    else:
        result = await tools.keyword_content_search(
            ["query"], task_id=task_id, url_part=".test", top_k=2
        )
    rows = orjson.loads(result)
    assert [row["chunk_id"] for row in rows] == [101, 102]
    assert all(row["doc_id"] == expected_doc for row in rows)
    assert all(row["task_id"] == expected_task for row in rows)
    expected_source = {
        10: ("https://original.test/a", "Original"),
        20: ("https://task.test/a", "Task source"),
        30: ("https://alias.test/a", "Alias source"),
    }[expected_doc]
    assert all((row["url"], row["title"]) == expected_source for row in rows)
    assert all("blake3" not in row for row in rows)


@pytest.mark.asyncio
@pytest.mark.parametrize("semantic", [False, True])
@pytest.mark.parametrize("by_doc", [False, True])
async def test_source_filters_are_applied_before_citation_selection(
    content_tools: tuple[PGTools, Lifecycle], semantic: bool, by_doc: bool
) -> None:
    tools, _ = content_tools
    doc_id = 30 if by_doc else None
    url_part = None if by_doc else "alias.test"
    if semantic:
        result = await tools.semantic_content_search(
            "query", browser_task_id=7, doc_id=doc_id, url_part=url_part
        )
    else:
        result = await tools.keyword_content_search(
            ["query"], task_id=7, doc_id=doc_id, url_part=url_part
        )
    rows = orjson.loads(result)
    assert len(rows) == 3
    assert all(row["doc_id"] == 30 and row["task_id"] == 7 for row in rows)
    assert all(row["url"] == "https://alias.test/a" for row in rows)


@pytest.mark.asyncio
@pytest.mark.parametrize("semantic", [False, True])
async def test_unscoped_url_filter_returns_that_sources_task(
    content_tools: tuple[PGTools, Lifecycle], semantic: bool
) -> None:
    tools, _ = content_tools
    if semantic:
        result = await tools.semantic_content_search("query", url_part="alias.test")
    else:
        result = await tools.keyword_content_search(["query"], url_part="alias.test")
    rows = orjson.loads(result)
    assert [row["chunk_id"] for row in rows] == [101, 102, 103]
    assert all(row["doc_id"] == 30 and row["task_id"] == 7 for row in rows)
    assert all(row["title"] == "Alias source" for row in rows)


@pytest.mark.asyncio
@pytest.mark.parametrize("semantic", [False, True])
async def test_task_scope_never_leaks_unassociated_sources(
    content_tools: tuple[PGTools, Lifecycle], semantic: bool
) -> None:
    tools, _ = content_tools
    for task_id, doc_id in [(7, 10), (8, 20), (9, 40), (9, 50)]:
        if semantic:
            result = await tools.semantic_content_search(
                "query", browser_task_id=task_id, doc_id=doc_id
            )
        else:
            result = await tools.keyword_content_search(
                ["query"], task_id=task_id, doc_id=doc_id
            )
        assert orjson.loads(result) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("semantic", [False, True])
async def test_text_retrieval_excludes_image_only_content(
    content_tools: tuple[PGTools, Lifecycle], semantic: bool
) -> None:
    tools, _ = content_tools
    if semantic:
        result = await tools.semantic_content_search("query", browser_task_id=10)
    else:
        result = await tools.keyword_content_search(["query"], task_id=10)
    assert orjson.loads(result) == []


@pytest.mark.asyncio
async def test_document_page_lookup_preserves_public_fields(
    content_tools: tuple[PGTools, Lifecycle],
) -> None:
    tools, _ = content_tools
    rows = orjson.loads(await tools.keyword_content_search(
        ["query"], task_id=7, doc_id=30, pages=[2]
    ))
    assert rows == [{
        "task_id": 7, "chunk_id": 103, "doc_id": 30, "page": 2,
        "chunk_index": 0, "text_chunk": "query third",
        "url": "https://alias.test/a", "title": "Alias source",
    }]
    assert orjson.loads(await tools.get_browser_task(7)) == {
        "starting_task": "duplicates"
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("task_id", "text_chunks", "image_chunks"),
    [(7, 3, 2), (9, 3, 2), (8, 1, 1), (10, 0, 1)],
)
async def test_progress_counts_shared_text_and_images_once(
    content_tools: tuple[PGTools, Lifecycle],
    task_id: int,
    text_chunks: int,
    image_chunks: int,
) -> None:
    _, service = content_tools
    progress = await service.browser_progress(task_id)
    assert progress.text_chunks == text_chunks
    assert progress.image_chunks == image_chunks
    assert progress.has_content
