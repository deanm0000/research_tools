from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import orjson
import pytest
from pgvector import Vector

from dean_research_tools.retriever import PGTools


@pytest.fixture
def browser_tools():
    tools = object.__new__(PGTools)
    tools.embeddings = SimpleNamespace(
        embed_search=AsyncMock(return_value=[0.1, 0.2]),
        embed_texts=AsyncMock(return_value=[[0.1, 0.2]]),
    )
    tools.settings = SimpleNamespace(queue_conn_str="dummy")
    cursor = AsyncMock()
    cursor.fetchall.return_value = []

    @asynccontextmanager
    async def get_cur():
        yield cursor

    tools._get_cur = get_cur
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
    result = await tools.semantic_content_search("query", top_k=3, **filters)
    query, values = cursor.execute.call_args.args
    sql = query.as_string()

    assert "FROM ai_proj_browser.text_content bc" in sql
    assert "INNER JOIN ai_proj_browser.docs bd using (doc_id)" in sql
    assert "INNER JOIN ai_proj_browser.tasks bt using (task_id)" in sql
    assert "bc.page" in sql
    assert "bc.chunk_indx AS chunk_index" in sql
    assert "ORDER BY bc.embedding <=> emb.embedding" in sql
    assert values[1:] == ([7, 9, "example", 3] if filtered else [3])
    if filtered:
        assert "AND bt.task_id = %s" in sql
        assert "AND bc.doc_id = %s" in sql
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
    assert "INNER JOIN ai_proj_browser.docs bd using (doc_id)" in sql
    assert "INNER JOIN ai_proj_browser.tasks bt using (task_id)" in sql
    assert "bc.chunk_indx AS chunk_index" in sql
    assert "AND bt.task_id = %s" in sql
    assert "ORDER BY bc.doc_id, bc.page, bc.chunk_indx" in sql
    assert values == (
        [["query"], 7, 9, [2], "example", 3] if filtered else [["query"], 7, 3]
    )
    if filtered:
        assert "AND bc.doc_id = %s" in sql
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
async def test_semantic_browser_task_query(browser_tools):
    tools, cursor = browser_tools
    await tools.semantic_browser_task_search("query", top_k=3)
    query, values = cursor.execute.call_args.args
    sql = query.as_string()

    assert "FROM ai_proj_browser.tasks bt" in sql
    assert "bt.task_id" in sql
    assert "ORDER BY bt.embedding <=> emb.embedding" in sql
    assert values[1:] == [3]


@pytest.mark.asyncio
@pytest.mark.parametrize("research_task_id", [None, 7])
async def test_queue_browser_query(browser_tools, research_task_id):
    tools, cursor = browser_tools
    cursor.fetchone.return_value = {"task_id": 42}
    vector = Vector([0.1, 0.2])
    with patch("dean_research_tools.retriever.Queue") as queue:
        queue.return_value.send_message = AsyncMock()
        await tools.queue_browser(
            research_task_id, {}, "objective", vector, deployment_id=11
        )
        sql, values = cursor.execute.call_args.args

        assert "INSERT INTO ai_proj_browser.tasks" in sql
        assert "(starting_task, embedding, deployment_id, research_task_id)" in sql
        assert "VALUES (%s, %s, %s, %s)" in sql
        assert "RETURNING task_id" in sql
        assert values == ("objective", vector, 11, research_task_id)
        queue.return_value.send_message.assert_awaited_once_with(
            {"function": "browser", "payload": 42}
        )
