from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest
from pgvector import Vector
from psycopg.pq import TransactionStatus

from dean_research_tools.retriever import PGTools


@pytest.fixture
def browser_tools():
    tools = object.__new__(PGTools)
    tools.browser_requests = []
    tools.browser_requests_started = 0
    tools.embeddings = SimpleNamespace(
        embed_search=AsyncMock(return_value=[0.1, 0.2]),
        embed_texts=AsyncMock(return_value=[[0.1, 0.2]]),
        embed_documents=AsyncMock(return_value=[[0.1, 0.2]]),
    )
    tools.settings = SimpleNamespace(queue_conn_str="dummy", queue_name="custom-worker")
    cursor = AsyncMock()
    cursor.fetchall.return_value = []

    @asynccontextmanager
    async def get_cur():
        yield cursor

    tools._get_cur = get_cur

    @asynccontextmanager
    async def get_conn() -> AsyncIterator[SimpleNamespace]:
        conn = SimpleNamespace(
            info=SimpleNamespace(transaction_status=TransactionStatus.IDLE),
            transaction=MagicMock(return_value=AsyncMock()),
        )
        yield conn

    tools._get_conn = get_conn
    return tools, cursor


@pytest.mark.asyncio
@pytest.mark.parametrize("filtered", [False, True])
async def test_semantic_content_query(browser_tools, filtered):
    tools, cursor = browser_tools
    filters = (
        {"browser_task_id": 7, "doc_id": 9, "url_part": "example"} if filtered else {}
    )
    cursor.fetchall.return_value = [
        {"task_id": 7, "doc_id": 9, "page": 2, "chunk_index": 3, "text_chunk": "text"}
    ]
    result = await tools.semantic_content_search(
        "query", top_k=3, min_score=0.7, **filters
    )
    query, values = cursor.execute.call_args.args
    sql = query.as_string()

    assert "FROM ai_proj_browser.text_content bc" in sql
    assert "INNER JOIN LATERAL" in sql
    assert "WHERE bd.blake3 = bc.blake3" in sql
    assert "ORDER BY bd.doc_id" in sql
    assert "ai_proj_browser.tasks bt" not in sql
    assert "bc.page" in sql
    assert "bc.chunk_indx AS chunk_index" in sql
    assert "ORDER BY bc.embedding <=> emb.embedding" in sql
    assert "(1 - (bc.embedding <=> emb.embedding))::double precision >= %s" in sql
    assert values[1:] == ([7, 9, "example", 0.7, 3] if filtered else [0.7, 3])
    if filtered:
        assert "7::bigint AS task_id" in sql
        assert "scope.task_id = %s AND scope.doc_id = bd.doc_id" in sql
        assert "AND bd.doc_id = %s" in sql
        assert "AND bd.url ILIKE '%%' || %s || '%%'" in sql
    assert orjson.loads(result) == cursor.fetchall.return_value


@pytest.mark.asyncio
@pytest.mark.parametrize("filtered", [False, True])
async def test_keyword_content_query(browser_tools, filtered):
    tools, cursor = browser_tools
    filters = {"doc_id": 9, "pages": [2], "url_part": "example"} if filtered else {}
    await tools.keyword_content_search(["query"], task_id=7, top_k=3, **filters)
    query, values = cursor.execute.call_args.args
    sql = query.as_string()

    assert "FROM ai_proj_browser.text_content bc" in sql
    assert "INNER JOIN LATERAL" in sql
    assert "WHERE bd.blake3 = bc.blake3" in sql
    assert "bc.chunk_indx AS chunk_index" in sql
    assert "FROM ai_proj_browser.tasks_docs scope" in sql
    assert "ORDER BY bd.doc_id, bc.page, bc.chunk_indx, bc.chunk_id" in sql
    assert values == (
        [7, 9, "example", ["query"], [2], 3] if filtered else [7, ["query"], 3]
    )
    if filtered:
        assert "AND bd.doc_id = %s" in sql
        assert "AND bc.page = ANY(%s)" in sql
        assert "AND bd.url ILIKE '%%' || %s || '%%'" in sql


@pytest.mark.asyncio
async def test_get_browser_task_query(browser_tools):
    tools, cursor = browser_tools
    cursor.fetchone.return_value = {"starting_task": "objective"}
    result = await tools.get_browser_task(7)
    sql, values = cursor.execute.call_args.args

    assert "FROM ai_proj_browser.tasks" in sql
    assert "WHERE task_id = %s" in sql
    assert values == [7]
    assert orjson.loads(result) == {"starting_task": "objective"}


@pytest.mark.asyncio
@pytest.mark.parametrize("semantic", [False, True])
async def test_task_scope_uses_shared_documents_without_join_duplicates(
    browser_tools: tuple[PGTools, AsyncMock], semantic: bool
) -> None:
    tools, cursor = browser_tools
    reused_chunk = {"task_id": 7, "doc_id": 9, "text_chunk": "reused"}
    cursor.fetchall.return_value = [reused_chunk]
    if semantic:
        result = await tools.semantic_content_search("query", browser_task_id=7)
    else:
        result = await tools.keyword_content_search(["query"], task_id=7)
    query, values = cursor.execute.call_args.args
    sql = query.as_string()
    assert "SELECT 1 FROM ai_proj_browser.tasks_docs scope" in sql
    assert "scope.task_id = %s AND scope.doc_id = bd.doc_id" in sql
    assert "JOIN ai_proj_browser.tasks_docs" not in sql
    assert values.count(7) == 1
    assert orjson.loads(result) == [reused_chunk]


@pytest.mark.asyncio
async def test_semantic_browser_task_query(browser_tools):
    tools, cursor = browser_tools
    await tools.semantic_browser_task_search("query", top_k=3, min_score=0.6)
    query, values = cursor.execute.call_args.args
    sql = query.as_string()

    assert "FROM ai_proj_browser.tasks bt" in sql
    assert "bt.task_id" in sql
    assert "ORDER BY bt.embedding <=> emb.embedding" in sql
    assert "(1 - (bt.embedding <=> emb.embedding))::double precision >= %s" in sql
    assert values[1:] == [0.6, 3]
    tools.embeddings.embed_search.assert_awaited_once_with("query")
    tools.embeddings.embed_documents.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("research_task_id", [None, 7])
async def test_queue_browser_query(browser_tools, research_task_id):
    tools, cursor = browser_tools
    cursor.fetchone.return_value = {"task_id": 42}
    vector = Vector([0.1, 0.2])
    with (
        patch("dean_research_tools.retriever.Queue") as queue,
        patch("dean_research_tools.retriever.Lifecycle") as lifecycle,
        patch(
            "dean_research_tools.retriever.ensure_pgvector_registered",
            new_callable=AsyncMock,
        ),
    ):
        queue.return_value.send_message = AsyncMock()
        lifecycle.return_value.create_browser = AsyncMock(return_value=42)
        assert await tools.queue_browser(
            research_task_id, {}, "objective", vector
        ) == 42
        lifecycle.return_value.create_browser.assert_awaited_once_with(
            research_task_id, {}, "objective", vector
        )
        queue.assert_called_once_with(conn_str="dummy", queue="custom-worker")
        queue.return_value.send_message.assert_awaited_once_with(
            {"function": "browser", "payload": 42}
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["semantic_content_search", "semantic_browser_task_search"])
@pytest.mark.parametrize("min_score", [-0.1, 1.1])
async def test_invalid_threshold_never_queries(
    browser_tools: tuple[PGTools, AsyncMock], method: str, min_score: float
) -> None:
    tools, cursor = browser_tools
    result = await getattr(tools, method)("query", min_score=min_score)
    assert result == "min_score must be between 0.0 and 1.0"
    cursor.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_queue_rejects_uncommitted_caller_transaction(
    browser_tools: tuple[PGTools, AsyncMock],
) -> None:
    tools, cursor = browser_tools

    @asynccontextmanager
    async def get_conn() -> AsyncIterator[SimpleNamespace]:
        yield SimpleNamespace(
            info=SimpleNamespace(transaction_status=TransactionStatus.INTRANS)
        )

    tools._get_conn = get_conn
    with patch("dean_research_tools.retriever.Queue") as queue:
        queue.return_value.send_message = AsyncMock()
        with pytest.raises(RuntimeError, match="idle connection"):
            await tools.queue_browser(None, {}, "objective", [0.1])
        queue.return_value.send_message.assert_not_awaited()
    cursor.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_browser_use_only_records_handoff(
    browser_tools: tuple[PGTools, AsyncMock],
) -> None:
    tools, cursor = browser_tools
    result = await tools.browser_use("objective")
    assert result == "Browser task triggered."
    assert tools.triggered_browser_use is not None
    assert tools.triggered_browser_use[0] == "objective"
    assert isinstance(tools.triggered_browser_use[1], Vector)
    tools.embeddings.embed_documents.assert_awaited_once_with(["objective"])
    tools.embeddings.embed_search.assert_not_awaited()
    cursor.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_queue_commits_before_dispatch(browser_tools: tuple[PGTools, AsyncMock]) -> None:
    tools, _ = browser_tools
    events: list[str] = []

    @asynccontextmanager
    async def transaction() -> AsyncIterator[None]:
        yield
        events.append("commit")

    @asynccontextmanager
    async def get_conn() -> AsyncIterator[SimpleNamespace]:
        yield SimpleNamespace(
            info=SimpleNamespace(transaction_status=TransactionStatus.IDLE),
            transaction=transaction,
        )

    async def create_browser(
        research_task_id: int | None,
        state: dict[str, str],
        objective: str,
        vector: list[float],
    ) -> int:
        events.append("create")
        return 42

    async def send_message(message: dict[str, str | int]) -> None:
        events.append("publish")

    tools._get_conn = get_conn
    with (
        patch("dean_research_tools.retriever.Queue") as queue,
        patch("dean_research_tools.retriever.Lifecycle") as lifecycle,
        patch("dean_research_tools.retriever.ensure_pgvector_registered", new_callable=AsyncMock),
    ):
        lifecycle.return_value.create_browser = create_browser
        queue.return_value.send_message = send_message
        assert await tools.queue_browser(None, {}, "objective", [0.1]) == 42
    assert events == ["create", "commit", "publish"]


@pytest.mark.asyncio
async def test_unscoped_search_reports_first_task_using_document(browser_tools):
    tools, cursor = browser_tools
    await tools.semantic_content_search("query", url_part="example")
    query, _ = cursor.execute.call_args.args
    sql = query.as_string()
    assert "SELECT min(td.task_id) FROM ai_proj_browser.tasks_docs td" in sql
    assert "WHERE td.doc_id = bd.doc_id) AS task_id" in sql
