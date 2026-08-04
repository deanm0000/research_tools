from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from inspect import ismethod
from typing import TYPE_CHECKING, Any, Literal, Sequence, get_args
from weakref import WeakSet

import orjson
from ddgs import DDGS
from ddgs.exceptions import DDGSException
from langchain_core.tools import StructuredTool
from pgvector import Vector
from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg.sql import SQL

import dean_research_tools.models as models
from dean_research_tools.config import SettingsLike, load_settings
from dean_research_tools.embeddings import EmbeddingsModel

if TYPE_CHECKING:
    from psycopg_pool import AsyncConnectionPool
_registered_pgvector: WeakSet[AsyncConnection] = WeakSet()


def _to_pascal_inputs(s: str) -> str:
    return "".join(word.capitalize() for word in s.split("_")) + "Input"


async def ensure_pgvector_registered(conn: AsyncConnection) -> None:
    if conn in _registered_pgvector:
        return
    await register_vector_async(conn)
    _registered_pgvector.add(conn)


AVAILABLE_TOOLS = Literal[
    "semantic_content_search",
    "keyword_content_search",
    "get_browser_task",
    "semantic_browser_task_search",
    "browser_use",
    "semantic_research_search",
    "get_research_result",
    "web_search",
]


class PGTools:
    def __init__(
        self,
        *,
        include: Sequence[AVAILABLE_TOOLS] | None = None,
        exclude: Sequence[AVAILABLE_TOOLS] | None = None,
        settings: SettingsLike | None = None,
        conn_or_pool: AsyncConnectionPool | AsyncConnection | None = None,
        # research_task_id: int,
    ):
        self.settings = load_settings(settings)
        self.include = include
        self.exclude = exclude

        self.conn_or_pool = conn_or_pool
        self._close_conn = conn_or_pool is None
        self.embeddings = EmbeddingsModel(settings=self.settings)
        # self.research_task_id = research_task_id
        self.triggered_browser_use: tuple[str, Vector] | None = None

    @asynccontextmanager
    async def _get_conn(self):
        try:
            from psycopg_pool import AsyncConnectionPool
        except ImportError:
            AsyncConnectionPool = None
        if self.conn_or_pool is None:
            self.conn_or_pool = await AsyncConnection.connect(
                self.settings.db_dsn.get_secret_value()
            )
            await ensure_pgvector_registered(self.conn_or_pool)
            await self.conn_or_pool.set_autocommit(True)
            yield self.conn_or_pool
        elif isinstance(self.conn_or_pool, AsyncConnection):
            yield self.conn_or_pool
        elif AsyncConnectionPool is not None and isinstance(
            self.conn_or_pool, AsyncConnectionPool
        ):
            async with self.conn_or_pool.connection() as conn:
                yield conn
        else:
            raise ValueError("could not make connection")

    @asynccontextmanager
    async def _get_cur(self):
        async with self._get_conn() as conn:
            await ensure_pgvector_registered(conn)
            async with conn.cursor(row_factory=dict_row) as cur:
                yield cur

    @asynccontextmanager
    async def _get_curt(self):
        async with self._get_conn() as conn:
            await ensure_pgvector_registered(conn)
            async with conn.cursor() as cur:
                yield cur

    def _get_all_tools(self) -> list[StructuredTool]:
        tools = []
        for tool_name in dir(self):
            if tool_name[0] == "_" or not ismethod(getattr(self, tool_name)):
                continue
            kwargs = {
                "coroutine": getattr(self, tool_name),
                "args_schema": getattr(models, _to_pascal_inputs(tool_name)),
                "description": getattr(self, tool_name).__doc__.replace(
                    "RETURN_DIRECT", ""
                ),
            }
            if "RETURN_DIRECT" in getattr(self, tool_name).__doc__:
                kwargs["return_direct"] = True
            tools.append(StructuredTool.from_function(**kwargs))
        return tools

    def _get_tools(
        self,
    ) -> list[StructuredTool]:
        tools_to_get: tuple[AVAILABLE_TOOLS] = tuple([])
        if self.include is not None and self.exclude is not None:
            raise ValueError("Cannot specify both include and exclude")
        elif self.include is None and self.exclude is None:
            tools_to_get: tuple[AVAILABLE_TOOLS] = get_args(AVAILABLE_TOOLS)
        elif self.include is not None:
            tools_to_get = tuple(
                x for x in get_args(AVAILABLE_TOOLS) if x in self.include
            )
        elif self.exclude is not None:
            tools_to_get = tuple(
                t for t in get_args(AVAILABLE_TOOLS) if t not in self.exclude
            )

        return [t for t in self._get_all_tools() if t.name in tools_to_get]

    async def __aenter__(self) -> list[StructuredTool]:
        return self._get_tools()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if isinstance(self.conn_or_pool, AsyncConnection) and self._close_conn:
            # if the user brings a pool or connection then they must close it
            await self.conn_or_pool.close()
            self.conn_or_pool = None

    async def _validate(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0,
    ):
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not (0.0 <= min_score <= 1.0):
            raise ValueError("min_score must be between 0.0 and 1.0")

    async def semantic_research_search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0,
    ) -> str:
        """Search for past research tasks using the semantic embedding of the query. It returns the research_task_id,
        starting_task, and the similarity score. From that information the result of research can be got with get_research_result tool."""
        await self._validate(query, top_k=top_k, min_score=min_score)

        query_embedding = await self.embeddings.embed_search(query)
        sql = SQL("""
            WITH emb AS (
                SELECT %s::vector AS embedding
                )
            SELECT
                id as research_task_id,
                starting_task,
                (1 - (bc.embedding <=> emb.embedding))::double precision AS score
            FROM ai_proj.researcher_tasks
            CROSS JOIN emb bc
            WHERE (1 - (bc.embedding <=> emb.embedding))::double precision >= %s
            ORDER BY bc.embedding <=> emb.embedding
            LIMIT %s
            """)
        values = [Vector(query_embedding), min_score, top_k]

        async with self._get_cur() as cur:
            await cur.execute(sql, values)
            res = await cur.fetchall()
            return orjson.dumps(res).decode("utf-8")

    async def get_research_result(self, research_task_id: int) -> str:
        """Get the result of a research task by its research_task_id. It returns the output of prior research. Use the semantic_research_search tool
        to find the research_task_id of prior research tasks."""
        sql = SQL("""
            select history->'messages'-> -1 ->>'content' as content from ai_proj.researcher_tasks_done
            WHERE id = %s
            """)
        values = [research_task_id]

        async with self._get_cur() as cur:
            await cur.execute(sql, values)
            res = await cur.fetchone()
            assert res is not None, (
                f"No research result found for research_task_id {research_task_id}"
            )
            return res["content"]

    async def semantic_content_search(
        self,
        query: str,
        task_id: int | None = None,
        url_part: str | None = None,
        doc_id: int | None = None,
        top_k: int = 5,
        min_score: float = 0,
    ) -> str:
        """Search for relevant content in the browser_content table using vector similarity."""
        await self._validate(query, top_k=top_k, min_score=min_score)

        query_embedding = await self.embeddings.embed_search(query)

        select = SQL("""
            WITH emb AS (
                SELECT %s::vector AS embedding
                )
            SELECT
                bt.id as task_id,
                bc.chunk_id,
                bc.doc_id,
                (bc.chunk_meta->>'page')::int AS page,
                (bc.chunk_meta->>'chunk_index')::int AS chunk_index,
                bc.text_chunk,
                COALESCE(bd.url, '') AS url,
                COALESCE(bd.title, '') AS title,
                (1 - (bc.embedding <=> emb.embedding))::double precision AS score
            FROM ai_proj.browser_content bc
            INNER JOIN ai_proj.browser_docs bd using (doc_id)
            INNER JOIN ai_proj.browser_tasks bt using (id)
            CROSS JOIN emb
            """)
        values: list[Vector | int | str] = [Vector(query_embedding)]
        wheres = [
            SQL("""
            WHERE bc.embedding IS NOT NULL
              AND bc.text_chunk IS NOT NULL
            """)
        ]

        if task_id is not None:
            wheres.append(SQL("AND bt.id = %s "))
            values.append(task_id)

        if doc_id is not None:
            wheres.append(SQL("AND bc.doc_id = %s "))
            values.append(doc_id)

        if url_part is not None:
            wheres.append(SQL("AND bd.url ILIKE '%%' || %s || '%%'"))
            values.append(url_part)

        limit = SQL("""
            ORDER BY bc.embedding <=> emb.embedding
            LIMIT %s
            """)
        values.append(top_k)
        sql = select + SQL(" ").join(wheres) + limit
        async with self._get_cur() as cur:
            await cur.execute(sql, values)
            res = await cur.fetchall()
            return orjson.dumps(res).decode("utf-8")

    async def keyword_content_search(
        self,
        keywords: list[str],
        task_id: int | None = None,
        url_part: str | None = None,
        doc_id: int | None = None,
        pages: list[int] | None = None,
        top_k: int = 5,
    ) -> str:
        """Search for relevant content in the browser_content table using keywords, it is case-insensitive.
        Must provide either a task_id or url_part to filter results."""
        if not keywords:
            raise ValueError("keywords must not be empty")
        if not task_id and not url_part:
            raise ValueError("Either task_id or url_part must be provided")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if pages and not doc_id:
            raise ValueError("doc_id must be provided if pages are specified")

        select = SQL("""
            SELECT
                bt.id as task_id,
                bc.chunk_id,
                bc.doc_id,
                (bc.chunk_meta->>'page')::int AS page,
                (bc.chunk_meta->>'chunk_index')::int AS chunk_index,
                bc.text_chunk,
                COALESCE(bd.url, '') AS url,
                COALESCE(bd.title, '') AS title
            FROM ai_proj.browser_content bc
            INNER JOIN ai_proj.browser_docs bd using (doc_id)
            INNER JOIN ai_proj.browser_tasks bt using (id)
            """)

        wheres = [
            SQL("""
            WHERE bc.text_chunk IS NOT NULL
              AND EXISTS (
                  SELECT 1
                  FROM unnest(%s::text[]) AS kw
                  WHERE bc.text_chunk ILIKE '%%' || kw || '%%'
              )
        """)
        ]
        values: list[str | int | list[str] | list[int]] = [keywords]

        if task_id is not None:
            wheres.append(SQL("AND bt.id = %s"))
            values.append(task_id)
        if doc_id is not None:
            wheres.append(SQL("AND bc.doc_id = %s"))
            values.append(doc_id)
            if pages:
                wheres.append(SQL("AND (bc.chunk_meta->>'page')::int = ANY(%s)"))
                values.append(pages)
        if url_part is not None:
            wheres.append(SQL("AND bd.url ILIKE '%%' || %s || '%%'"))
            values.append(url_part)

        limit = SQL("""
                    ORDER BY doc_id, (bc.chunk_meta->>'page')::int, (bc.chunk_meta->>'chunk_index')::int
                    LIMIT %s
                    """)
        values.append(top_k)
        sql = select + SQL(" ").join(wheres) + limit

        async with self._get_cur() as cur:
            await cur.execute(sql, values)
            res = await cur.fetchall()
            return orjson.dumps(res).decode("utf-8")

    async def get_browser_task(self, task_id: int) -> str:
        """Retrieve the starting_task for a given task_id from the browser_tasks table.
        This is the same info from semantic_task_search."""

        sql = """
            SELECT starting_task
            FROM ai_proj.browser_tasks
            WHERE id = %s
        """
        async with self._get_cur() as cur:
            await cur.execute(sql, [task_id])
            res = await cur.fetchone()
            return orjson.dumps(res).decode("utf-8")

    async def semantic_browser_task_search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0,
    ) -> str:
        """Search for relevant tasks in the browser_tasks table using vector similarity. Your query should
        be a natural language description of the task as if you were instructing the browser subagent to get new information.
        """
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not (0.0 <= min_score <= 1.0):
            raise ValueError("min_score must be between 0.0 and 1.0")

        query_embedding = await self.embeddings.embed_texts(
            [query], input_type="search_query"
        )
        if query_embedding is None:
            raise ValueError("Failed to generate query embedding")
        query_embedding = query_embedding[0]

        sql = SQL("""
            WITH emb AS (
                SELECT %s::vector AS embedding
                )
            SELECT
                bt.id as task_id,
                bt.starting_task,
                (1 - (bt.embedding <=> emb.embedding))::double precision AS score
            FROM ai_proj.browser_tasks bt
            CROSS JOIN emb
            ORDER BY bt.embedding <=> emb.embedding
            LIMIT %s
            """)
        values = [Vector(query_embedding), top_k]
        async with self._get_cur() as cur:
            await cur.execute(sql, values)
            res = await cur.fetchall()
            return orjson.dumps(res).decode("utf-8")

    async def web_search(self, query: str, max_results: int = 5) -> str:
        """Search the web for URLs and metadata associated with the query. Use this to find official sites that
        the browser_use tool can visit."""
        cleaned_query = query.strip()
        if len(cleaned_query) == 0:
            return "No query provided for web search."

        bounded_max_results = min(max(max_results, 1), 10)

        try:
            raw_results = await asyncio.to_thread(
                _run_ddgs_text_search,
                cleaned_query,
                bounded_max_results,
            )
        except DDGSException as exc:
            if "No results found" in str(exc):
                return f"No web results found for '{cleaned_query}'."
            return f"Web search failed: {exc}"

        if len(raw_results) == 0:
            return f"No web results found for '{cleaned_query}'."

        return orjson.dumps(raw_results).decode("utf-8")

    async def browser_use(self, objective: str) -> str:
        """Browser-based ingestion agent. Given an objective, it navigates the
        web with a real browser and ingests content based on your objective into the
        vector database for later retrieval. Returns a task_id you can use to query
        the ingested content once ingestion completes.

        Before using this tool, try to find official websites using the web_search
        tool first, then call browser_use with a specific url. If you want to ingest
        content from multiple sites, call browser_use multiple times (one per site).

        This agent ONLY collects and stores content — it does not read, summarize,
        analyze, or report on what it finds. Do not phrase the objective as a
        request for a report, summary, or answer (e.g. avoid "find and summarize
        X"). Instead, phrase it as a content-gathering directive, e.g.:
        "Go to www.example.com, navigate the site, and ingest all content related
        to ..."

        After ingestion, query the vector database using the returned task_id to
        retrieve and reason over the content yourself.RETURN_DIRECT"""

        query_embedding = await self.embeddings.embed_texts(
            [objective], input_type="search_query"
        )
        if query_embedding is None:
            raise ValueError("Failed to generate query embedding")
        query_embedding = query_embedding[0]
        self.triggered_browser_use = (objective, Vector(query_embedding))
        return "Browser task triggered."


def _run_ddgs_text_search(query: str, max_results: int) -> list[dict[str, Any]]:
    with DDGS() as client:
        # ddgs currently returns mapping-like result rows.
        return client.text(
            query,
            max_results=max_results,
            backend=["brave", "duckduckgo", "google", "yahoo"],
        )
