from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pgvector import Vector

from dean_research_tools.lifecycle import (
    BrowserCompletion,
    BrowserContinuation,
    BrowserDispatch,
    BrowserProgress,
    Lifecycle,
    ResearchContinuation,
    ResearchFinish,
    to_json_str,
)


@pytest.fixture
def lifecycle() -> tuple[Lifecycle, AsyncMock]:
    cur = AsyncMock()
    manager = MagicMock()
    manager.__aenter__ = AsyncMock(return_value=cur)
    manager.__aexit__ = AsyncMock(return_value=None)
    conn = MagicMock()
    conn.cursor.return_value = manager
    return Lifecycle(conn), cur


def test_state_serializes_message_models() -> None:
    state = {"messages": [HumanMessage(content="question"), AIMessage(content="answer")]}
    restored = orjson.loads(to_json_str(state))
    assert restored["messages"][0]["type"] == "human"
    assert restored["messages"][1]["content"] == "answer"
    assert orjson.loads(to_json_str(restored)) == restored


@pytest.mark.asyncio
@pytest.mark.parametrize("research_task_id", [None, 7])
async def test_create_browser(lifecycle: tuple[Lifecycle, AsyncMock], research_task_id: int | None) -> None:
    service, cur = lifecycle
    cur.fetchone.return_value = {"task_id": 42}
    state = {"messages": [HumanMessage(content="question")]}
    assert await service.create_browser(research_task_id, state, "objective", [0.1, 0.2]) == 42
    sql, values = cur.execute.call_args.args
    if research_task_id is not None:
        assert "ai_proj.research_make_browser_task" in sql
        assert "::jsonb" in sql
        assert values[0] == 7
        assert orjson.loads(values[1])["messages"][0]["content"] == "question"
        assert values[2] == "objective"
    else:
        assert "INSERT INTO ai_proj_browser.tasks" in sql
        assert "ai_proj.current_func_settings" in sql
        assert "function_name = 'browser'" in sql
        assert "RETURNING task_id" in sql
        assert values[0] == "objective"
    assert isinstance(values[-1], Vector)


@pytest.mark.asyncio
async def test_missing_browser_settings(lifecycle: tuple[Lifecycle, AsyncMock]) -> None:
    service, cur = lifecycle
    cur.fetchone.return_value = None
    with pytest.raises(RuntimeError, match="current browser settings"):
        await service.create_browser(None, {}, "objective", [0.1])


@pytest.mark.asyncio
async def test_start_research(lifecycle: tuple[Lifecycle, AsyncMock]) -> None:
    service, cur = lifecycle
    cur.fetchone.return_value = {
        "starting_task": "question",
        "state": {"messages": []},
        "sys_prompt": "system",
        "model": "gpt",
    }
    task = await service.start_research(7)
    assert task.starting_task == "question"
    assert task.state == {"messages": []}
    assert task.sys_prompt == "system"
    assert task.model == "gpt"
    assert cur.execute.call_args.args == (
        "SELECT * FROM ai_proj.start_research_task(%s::integer)", (7,)
    )


@pytest.mark.asyncio
async def test_start_browser(lifecycle: tuple[Lifecycle, AsyncMock]) -> None:
    service, cur = lifecycle
    cur.fetchone.return_value = {"starting_task": "objective", "model": "gpt"}
    task = await service.start_browser(8)
    assert task.starting_task == "objective"
    assert task.model == "gpt"


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["start_research", "start_browser"])
async def test_start_missing(lifecycle: tuple[Lifecycle, AsyncMock], method: str) -> None:
    service, cur = lifecycle
    cur.fetchone.return_value = None
    with pytest.raises(LookupError):
        await getattr(service, method)(8)


def test_link_readiness_is_not_browser_finalization() -> None:
    progress = BrowserProgress(task_id=7, total_links=2, done_links=1, errored_links=1)
    assert progress.links_terminal
    assert not progress.finished
    assert not progress.has_content
    progress.status = "complete"
    progress.outcome = "partial"
    progress.text_chunks = 1
    assert progress.finished
    assert progress.has_content
    progress.pending_links = 1
    assert not progress.links_terminal


@pytest.mark.asyncio
@pytest.mark.parametrize("system_prompt", [None, "custom instructions"])
async def test_create_research(
    lifecycle: tuple[Lifecycle, AsyncMock], system_prompt: str | None
) -> None:
    service, cur = lifecycle
    cur.fetchone.return_value = {"id": 13}
    assert await service.create_research("question", "user", [0.1], system_prompt) == 13
    sql, values = cur.execute.call_args.args
    assert "ai_proj.current_func_settings" in sql
    assert "function_name = 'researcher'" in sql
    assert "INSERT INTO ai_proj.research_system_prompts" in sql
    assert "ORDER BY added DESC, sys_id DESC" in sql
    assert "INSERT INTO ai_proj.researcher_tasks" in sql
    assert values[:3] == (system_prompt, "question", "user")
    assert isinstance(values[3], Vector)


@pytest.mark.asyncio
async def test_missing_research_configuration(lifecycle: tuple[Lifecycle, AsyncMock]) -> None:
    service, cur = lifecycle
    cur.fetchone.return_value = None
    with pytest.raises(RuntimeError, match="current researcher settings"):
        await service.create_research("question", "user", [0.1])


@pytest.mark.asyncio
async def test_browser_progress(lifecycle: tuple[Lifecycle, AsyncMock]) -> None:
    service, cur = lifecycle
    cur.fetchone.return_value = {
        "task_id": 8,
        "status": "in_progress",
        "total_links": 3,
        "pending_links": 1,
        "done_links": 1,
        "errored_links": 1,
        "pending_research_ids": [],
    }
    progress = await service.browser_progress(8)
    assert progress.queued == 1
    assert progress.done == 1
    assert progress.errored == 1
    assert not progress.links_terminal
    sql, values = cur.execute.call_args.args
    assert "ai_proj_browser.links" in sql
    assert "WHERE l.status = 'queued'" in sql
    assert "ai_proj_browser.tasks_done" in sql
    assert "ai_proj.research_task_pause" in sql
    assert "b.publisher_task_id=t.task_id" in sql
    assert "b.published_at IS NULL AND b.resumed_at IS NULL" in sql
    assert sql.count("AND d.blake3 = c.blake3") == 2
    assert sql.count("WHERE scope.task_id = t.task_id") == 2
    assert "JOIN ai_proj_browser.tasks_docs d USING (doc_id)" not in sql
    assert values == (8,)


@pytest.mark.asyncio
async def test_browser_progress_missing(lifecycle: tuple[Lifecycle, AsyncMock]) -> None:
    service, cur = lifecycle
    cur.fetchone.return_value = None
    with pytest.raises(LookupError):
        await service.browser_progress(8)


@pytest.mark.asyncio
async def test_finish_browser_fetches_normalized_history(
    lifecycle: tuple[Lifecycle, AsyncMock],
) -> None:
    service, cur = lifecycle
    cur.fetchone.return_value = {
        "task_id": 8,
        "status": "complete",
        "outcome": "partial",
        "history": {"status": "partial"},
        "ingestion_summary": {"links_errored": 1},
        "errored_links": 1,
        "pending_research_ids": [13],
    }
    progress = await service.finish_browser(8, {"status": "completed"}, {"tokens": 7})
    assert progress.outcome == "partial"
    assert progress.history == {"status": "partial"}
    assert progress.ingestion_summary == {"links_errored": 1}
    assert progress.pending_research_ids == [13]
    assert progress.finished
    assert progress.links_terminal
    sql, values = cur.execute.call_args_list[0].args
    assert "ai_proj.finish_browser_task" in sql
    assert values[0] == 8
    assert orjson.loads(values[1]) == {"status": "completed"}
    assert orjson.loads(values[2]) == {"tokens": 7}
    assert cur.execute.await_count == 2


@pytest.mark.asyncio
async def test_browser_completion_recovery(lifecycle: tuple[Lifecycle, AsyncMock]) -> None:
    service, cur = lifecycle
    cur.fetchone.return_value = {
        "task_id": 8,
        "status": "complete",
        "history": {"status": "partial", "errors": ["missing source"]},
        "usage": {"total_tokens": 7},
    }
    result = await service.get_browser_completion(8)
    assert isinstance(result, BrowserCompletion)
    assert result.history["status"] == "partial"
    assert result.result == result.history
    assert result.usage == {"total_tokens": 7}
    sql, values = cur.execute.call_args.args
    assert "ai_proj_browser.tasks_done" in sql
    assert values == (8,)
    assert "start_browser_task" not in sql


@pytest.mark.asyncio
async def test_browser_completion_missing(lifecycle: tuple[Lifecycle, AsyncMock]) -> None:
    service, cur = lifecycle
    cur.fetchone.return_value = None
    assert await service.get_browser_completion(8) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("report", [None, "final report"])
async def test_finish_research(lifecycle: tuple[Lifecycle, AsyncMock], report: str | None) -> None:
    service, cur = lifecycle
    cur.fetchone.return_value = {
        "final_status": "complete", "parent_research_id": None, "batch_id": None,
        "reports_saved": 0,
    }
    state = {"messages": [AIMessage(content="answer")]}
    result = await service.finish_research(13, state, {"tokens": 7}, report, summary="Title")
    assert result == ResearchFinish(final_status="complete")
    sql, values = cur.execute.call_args.args
    assert "ai_proj.finish_research_task" in sql
    assert "finished AS MATERIALIZED" in sql
    assert "INSERT INTO ai_proj.researcher_reports" in sql
    assert "max(r.revision) + 1" in sql
    assert values[:2] == (13, 13)
    history = orjson.loads(values[2])
    assert history["messages"][0]["content"] == "answer"
    assert history["usage"] == {"tokens": 7}
    assert "usage" not in state
    assert values[3:] == ("Title", "complete", report, report)
    assert cur.execute.await_count == 1


@pytest.mark.asyncio
async def test_finish_research_returns_parent_continuation(
    lifecycle: tuple[Lifecycle, AsyncMock],
) -> None:
    service, cur = lifecycle
    cur.fetchone.return_value = {
        "final_status": "incomplete", "parent_research_id": 5, "batch_id": 9,
        "reports_saved": 1,
    }
    result = await service.finish_research(
        13, {"messages": []}, report="partial", terminal_status="incomplete",
    )
    assert result.final_status == "incomplete"
    assert result.continuation == ResearchContinuation(batch_id=9, research_id=5)
    assert isinstance(result.continuation, BrowserContinuation)
    assert cur.execute.call_args.args[1][4] == "incomplete"


@pytest.mark.asyncio
async def test_finish_research_optional_usage(lifecycle: tuple[Lifecycle, AsyncMock]) -> None:
    service, cur = lifecycle
    cur.fetchone.return_value = {
        "final_status": "complete", "parent_research_id": None, "batch_id": None,
        "reports_saved": 0,
    }
    await service.finish_research(13, {"messages": []})
    assert orjson.loads(cur.execute.call_args.args[1][2]) == {"messages": []}


@pytest.mark.asyncio
@pytest.mark.parametrize("finished", [False, True])
async def test_pending_browser_dispatch(
    lifecycle: tuple[Lifecycle, AsyncMock], finished: bool
) -> None:
    service, cur = lifecycle
    cur.fetchall.return_value = [{
        "task_id": 8, "status": "complete" if finished else "queued", "finished": finished
    }]
    result = await service.get_pending_browser(13)
    assert isinstance(result, BrowserDispatch)
    assert result.task_id == 8
    assert result.finished == finished
    sql, values = cur.execute.call_args.args
    assert "research_browser_members" in sql
    assert "LEFT JOIN ai_proj_browser.tasks_done" in sql
    assert "(d.task_id IS NOT NULL) AS finished" in sql
    assert values == (13,)


@pytest.mark.asyncio
async def test_no_pending_browser(lifecycle: tuple[Lifecycle, AsyncMock]) -> None:
    service, cur = lifecycle
    cur.fetchall.return_value = []
    assert await service.get_pending_browser(13) is None


@pytest.mark.asyncio
async def test_mark_research_error_only_changes_in_progress(
    lifecycle: tuple[Lifecycle, AsyncMock],
) -> None:
    service, cur = lifecycle
    await service.mark_research_error(13)
    sql, values = cur.execute.call_args.args
    assert "UPDATE ai_proj.researcher_tasks SET status = 'error'" in sql
    assert "WHERE id = %s AND status = 'in_progress'" in sql
    assert values == (13,)
    assert cur.execute.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("complete", [False, True])
async def test_is_research_complete(
    lifecycle: tuple[Lifecycle, AsyncMock], complete: bool
) -> None:
    service, cur = lifecycle
    cur.fetchone.return_value = {"complete": complete}
    assert await service.is_research_complete(13) == complete
    sql, values = cur.execute.call_args.args
    assert "SELECT EXISTS" in sql
    assert "status IN ('complete', 'incomplete')" in sql
    assert values == (13,)


@pytest.mark.asyncio
async def test_research_completion_inspection_failure(
    lifecycle: tuple[Lifecycle, AsyncMock],
) -> None:
    service, cur = lifecycle
    cur.fetchone.return_value = None
    with pytest.raises(RuntimeError, match="Could not inspect research task"):
        await service.is_research_complete(13)
