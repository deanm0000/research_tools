import asyncio
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest

from dean_research_tools import BrowserRequest, Lifecycle, PGTools
from dean_research_tools.config import Settings


@pytest.mark.asyncio
async def test_concurrent_browser_calls_preserve_every_request() -> None:
    tools = PGTools(settings=Settings.dummies())
    second_finished = asyncio.Event()

    async def embed(texts: list[str]) -> list[list[float]]:
        if texts == ["first"]:
            await second_finished.wait()
        else:
            second_finished.set()
        return [[0.1, 0.2]]

    tools.embeddings.embed_documents = AsyncMock(side_effect=embed)
    await asyncio.gather(tools.browser_use("first"), tools.browser_use("second"))
    assert tools.browser_requests_started == 2
    assert [request.objective for request in tools.browser_requests] == ["second", "first"]
    assert [
        request.objective for request in sorted(tools.browser_requests, key=lambda r: r.ordinal)
    ] == ["first", "second"]


@pytest.mark.asyncio
async def test_repeated_objectives_remain_distinct() -> None:
    tools = PGTools(settings=Settings.dummies())
    tools.embeddings.embed_documents = AsyncMock(return_value=[[0.1]])
    await asyncio.gather(tools.browser_use("same"), tools.browser_use("same"))
    assert sorted(request.ordinal for request in tools.browser_requests) == [0, 1]


@pytest.mark.asyncio
async def test_failed_preparation_is_detectable() -> None:
    tools = PGTools(settings=Settings.dummies())
    tools.embeddings.embed_documents = AsyncMock(side_effect=[
        [[0.1]], RuntimeError("embedding unavailable"),
    ])
    with pytest.raises(RuntimeError, match="embedding unavailable"):
        await asyncio.gather(tools.browser_use("first"), tools.browser_use("second"))
    assert tools.browser_requests_started == 2
    assert len(tools.browser_requests) == 1


@pytest.mark.asyncio
async def test_batch_creation_serializes_requests_and_live_state() -> None:
    cur = AsyncMock()
    cur.fetchall.return_value = [(41,), (42,)]
    manager = MagicMock()
    manager.__aenter__.return_value = cur
    conn = MagicMock()
    conn.cursor.return_value = manager
    lifecycle = Lifecycle(conn)
    requests = [BrowserRequest(objective="first", embedding=[0.1], ordinal=0)]
    assert await lifecycle.create_browser_batch(7, {"messages": []}, requests) == [41, 42]
    sql, values = cur.execute.call_args.args
    assert "research_make_browser_batch" in sql
    assert orjson.loads(values[2])[0]["objective"] == "first"
    with pytest.raises(ValueError, match="at least one"):
        await lifecycle.create_browser_batch(7, {"messages": []}, [])


@pytest.mark.asyncio
async def test_continuation_lookup_and_acknowledgment() -> None:
    cur = AsyncMock()
    cur.fetchone.return_value = {"batch_id": 9, "research_id": 7}
    manager = MagicMock()
    manager.__aenter__.return_value = cur
    conn = MagicMock()
    conn.cursor.return_value = manager
    lifecycle = Lifecycle(conn)
    continuation = await lifecycle.get_browser_continuation(42)
    assert continuation is not None and continuation.batch_id == 9
    assert cur.execute.call_args.args[1] == (42,)
    await lifecycle.acknowledge_browser_continuation(9)
    assert "SET published_at=now()" in cur.execute.call_args.args[0]
    cur.fetchone.return_value = None
    assert await lifecycle.get_browser_continuation(42) is None
