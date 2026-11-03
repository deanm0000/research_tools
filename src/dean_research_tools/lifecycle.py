"""Trusted lifecycle operations with caller-owned connections and transactions."""

from __future__ import annotations

from datetime import datetime
from typing import Any, TypeAlias

from pgvector import Vector
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

# LangChain state contains arbitrary message/tool values, including BaseMessage
# instances. TypeAdapter serializes these models; JSON-only types reject live state.
State: TypeAlias = dict[str, Any]
_state_adapter = TypeAdapter(State)


def to_json_str(state: State) -> str:
    """Serialize live LangChain messages as well as already serialized state."""
    return _state_adapter.dump_json(state).decode("utf-8")


class ResearchStart(BaseModel):
    model_config = ConfigDict(extra="allow")

    starting_task: str
    state: dict[str, JsonValue] | str | None = None
    sys_prompt: str
    model: str


class BrowserStart(BaseModel):
    model_config = ConfigDict(extra="allow")

    starting_task: str
    model: str


class BrowserCompletion(BaseModel):
    task_id: int
    status: str
    history: dict[str, JsonValue]
    usage: dict[str, JsonValue]
    finished_at: datetime | None = None

    @property
    def result(self) -> dict[str, JsonValue]:
        return self.history


class BrowserDispatch(BaseModel):
    task_id: int
    status: str
    finished: bool


class BrowserProgress(BaseModel):
    task_id: int
    status: str | None = None
    outcome: str | None = None
    total_links: int = 0
    pending_links: int = 0
    done_links: int = 0
    errored_links: int = 0
    text_chunks: int = 0
    image_chunks: int = 0
    errors: list[JsonValue] = Field(default_factory=list)
    ingestion_summary: dict[str, JsonValue] | None = None
    history: dict[str, JsonValue] | None = None
    pending_research_ids: list[int] = Field(default_factory=list)

    @property
    def queued(self) -> int:
        return self.pending_links

    @property
    def done(self) -> int:
        return self.done_links

    @property
    def errored(self) -> int:
        return self.errored_links

    @property
    def links_terminal(self) -> bool:
        return self.pending_links == 0

    @property
    def finished(self) -> bool:
        return self.status in ("complete", "error")

    @property
    def has_content(self) -> bool:
        return self.text_chunks > 0 or self.image_chunks > 0


class Lifecycle:
    """Database lifecycle service. Does not connect, commit, close, or enqueue."""

    def __init__(self, conn: AsyncConnection) -> None:
        self.conn = conn

    async def get_pending_browser(self, research_id: int) -> BrowserDispatch | None:
        """Recover an existing browser handoff after queue publication fails."""
        async with self.conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT bt.task_id, bt.status::text AS status,
                    (td.task_id IS NOT NULL) AS finished
                FROM ai_proj.researcher_tasks rt
                JOIN ai_proj.research_task_pause p ON p.research_id = rt.id
                JOIN ai_proj_browser.tasks bt ON bt.task_id = p.browser_task_id
                LEFT JOIN ai_proj_browser.tasks_done td ON td.task_id = bt.task_id
                WHERE rt.id = %s AND rt.status = 'paused_for_browser'
                """,
                (research_id,),
            )
            row = await cur.fetchone()
        return None if row is None else BrowserDispatch.model_validate(row)

    async def mark_research_error(self, research_id: int) -> None:
        """Make interrupted execution retryable without changing paused/completed work."""
        async with self.conn.cursor() as cur:
            await cur.execute(
                """
                UPDATE ai_proj.researcher_tasks SET status = 'error'
                WHERE id = %s AND status = 'in_progress'
                """,
                (research_id,),
            )

    async def is_research_complete(self, research_id: int) -> bool:
        """Check committed completion so duplicate continuations can be acknowledged."""
        async with self.conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1 FROM ai_proj.researcher_tasks_done WHERE id = %s
                ) AS complete
                """,
                (research_id,),
            )
            row = await cur.fetchone()
        if row is None:
            raise RuntimeError(f"Could not inspect research task {research_id}")
        return bool(row["complete"])

    async def create_research(
        self,
        question: str,
        user_id: str,
        embedding: list[float] | Vector,
        system_prompt: str | None = None,
    ) -> int:
        """Create queued research with the configured deployment and selected prompt.

        An explicit prompt creates a prompt record; otherwise the newest saved
        prompt is used. The CTE keeps prompt creation and task insertion atomic.
        """
        vector = embedding if isinstance(embedding, Vector) else Vector(embedding)
        async with self.conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                WITH params AS (SELECT %s::text AS system_prompt),
                settings AS (
                    SELECT deployment_id FROM ai_proj.current_func_settings
                    WHERE function_name = 'researcher'
                ),
                new_prompt AS (
                    INSERT INTO ai_proj.research_system_prompts (sys_prompt)
                    SELECT params.system_prompt FROM params CROSS JOIN settings
                    WHERE params.system_prompt IS NOT NULL
                    RETURNING sys_id
                ),
                chosen_prompt AS (
                    SELECT sys_id FROM new_prompt
                    UNION ALL
                    (SELECT sys_id FROM ai_proj.research_system_prompts
                     WHERE (SELECT system_prompt FROM params) IS NULL
                     ORDER BY added DESC, sys_id DESC LIMIT 1)
                )
                INSERT INTO ai_proj.researcher_tasks
                    (starting_task, user_id, embedding, sys_id, deployment_id)
                SELECT %s, %s, %s::vector, p.sys_id, s.deployment_id
                FROM chosen_prompt p CROSS JOIN settings s
                RETURNING id
                """,
                (system_prompt, question, user_id, vector),
            )
            row = await cur.fetchone()
        if row is None:
            raise RuntimeError(
                "Research task not created; check current researcher settings "
                "and saved system prompts"
            )
        return int(row["id"])

    async def create_browser(
        self,
        research_task_id: int | None,
        state: State,
        objective: str,
        vector: list[float] | Vector,
    ) -> int:
        """Create browser work, atomically saving state for research handoffs."""
        embedding = vector if isinstance(vector, Vector) else Vector(vector)
        async with self.conn.cursor(row_factory=dict_row) as cur:
            if research_task_id is not None:
                await cur.execute(
                    """
                    SELECT browser_task_id AS task_id
                    FROM ai_proj.research_make_browser_task(
                        %s::bigint, %s::jsonb, %s::text, %s::vector
                    )
                    """,
                    (research_task_id, to_json_str(state), objective, embedding),
                )
            else:
                await cur.execute(
                    """
                    INSERT INTO ai_proj_browser.tasks
                        (starting_task, embedding, deployment_id)
                    SELECT %s, %s::vector, deployment_id
                    FROM ai_proj.current_func_settings
                    WHERE function_name = 'browser'
                    RETURNING task_id
                    """,
                    (objective, embedding),
                )
            row = await cur.fetchone()
        if row is None or row["task_id"] is None:
            raise RuntimeError("Browser task not created; check current browser settings")
        return int(row["task_id"])

    async def start_research(self, research_task_id: int) -> ResearchStart:
        async with self.conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT * FROM ai_proj.start_research_task(%s::integer)",
                (research_task_id,),
            )
            row = await cur.fetchone()
        if row is None:
            raise LookupError(f"Research task {research_task_id} cannot be started")
        return ResearchStart.model_validate(row)

    async def start_browser(self, browser_task_id: int) -> BrowserStart:
        async with self.conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT * FROM ai_proj.start_browser_task(%s::integer)",
                (browser_task_id,),
            )
            row = await cur.fetchone()
        if row is None:
            raise LookupError(f"Browser task {browser_task_id} cannot be started")
        return BrowserStart.model_validate(row)

    async def finish_browser(
        self,
        browser_task_id: int,
        history: State,
        usage: dict[str, JsonValue],
    ) -> BrowserProgress:
        """Finalize with SQL's normalized outcome, then fetch the persisted result."""
        async with self.conn.cursor() as cur:
            await cur.execute(
                """
                SELECT ai_proj.finish_browser_task(
                    %s::integer, %s::jsonb, %s::jsonb
                )
                """,
                (browser_task_id, to_json_str(history), to_json_str(usage)),
            )
        return await self.browser_progress(browser_task_id)

    async def get_browser_completion(
        self, browser_task_id: int
    ) -> BrowserCompletion | None:
        """Recover committed browser results without rerunning browser automation."""
        async with self.conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT td.task_id, t.status::text AS status,
                    COALESCE(td.history, '{}'::jsonb) AS history,
                    COALESCE(td.usage, '{}'::jsonb) AS usage, td.finished_at
                FROM ai_proj_browser.tasks_done td
                JOIN ai_proj_browser.tasks t USING (task_id)
                WHERE td.task_id = %s
                """,
                (browser_task_id,),
            )
            row = await cur.fetchone()
        return None if row is None else BrowserCompletion.model_validate(row)

    async def browser_progress(self, browser_task_id: int) -> BrowserProgress:
        """Fetch current ingestion counts and the stored, normalized outcome."""
        async with self.conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT t.task_id, t.status::text AS status,
                    td.history->>'status' AS outcome, td.history,
                    COALESCE(td.history->'errors', '[]'::jsonb) AS errors,
                    td.history->'ingestion_summary' AS ingestion_summary,
                    counts.total_links, counts.pending_links,
                    counts.done_links, counts.errored_links,
                    (SELECT count(*) FROM ai_proj_browser.text_content c
                     JOIN ai_proj_browser.docs d USING (doc_id)
                     WHERE d.task_id = t.task_id) AS text_chunks,
                    (SELECT count(*) FROM ai_proj_browser.image_content c
                     JOIN ai_proj_browser.docs d USING (doc_id)
                     WHERE d.task_id = t.task_id) AS image_chunks,
                    ARRAY(
                        SELECT p.research_id FROM ai_proj.research_task_pause p
                        JOIN ai_proj.researcher_tasks rt ON rt.id = p.research_id
                        WHERE p.browser_task_id = t.task_id
                          AND rt.status = 'paused_for_browser'
                          AND td.task_id IS NOT NULL
                          AND counts.pending_links = 0
                        ORDER BY p.research_id
                    ) AS pending_research_ids
                FROM ai_proj_browser.tasks t
                LEFT JOIN ai_proj_browser.tasks_done td USING (task_id)
                CROSS JOIN LATERAL (
                    SELECT count(*) AS total_links,
                        count(*) FILTER (WHERE l.status = 'queued') AS pending_links,
                        count(*) FILTER (WHERE l.status = 'done') AS done_links,
                        count(*) FILTER (WHERE l.status = 'errored') AS errored_links
                    FROM ai_proj_browser.links l WHERE l.task_id = t.task_id
                ) counts
                WHERE t.task_id = %s
                """,
                (browser_task_id,),
            )
            row = await cur.fetchone()
        if row is None:
            raise LookupError(f"Browser task {browser_task_id} does not exist")
        return BrowserProgress.model_validate(row)

    async def finish_research(
        self,
        research_task_id: int,
        state: State,
        usage: dict[str, JsonValue] | None = None,
        report: str | None = None,
        *,
        summary: str | None = None,
    ) -> None:
        """Save completion and an optional report atomically, without LLM calls."""
        completed_state = state if usage is None else {**state, "usage": usage}
        async with self.conn.cursor() as cur:
            await cur.execute(
                """
                WITH finished AS MATERIALIZED (
                    SELECT %s::bigint AS id,
                        ai_proj.finish_research_task(
                            %s::integer, %s::jsonb, %s::text
                        ) AS result
                ),
                report_saved AS (
                    INSERT INTO ai_proj.researcher_reports (id, revision, content)
                    SELECT f.id, COALESCE(
                        (SELECT max(r.revision) + 1
                         FROM ai_proj.researcher_reports r WHERE r.id = f.id), 1
                    ), %s::text
                    FROM finished f WHERE %s::text IS NOT NULL
                    RETURNING id
                )
                SELECT f.id, (SELECT count(*) FROM report_saved) FROM finished f
                """,
                (
                    research_task_id,
                    research_task_id,
                    to_json_str(completed_state),
                    summary,
                    report,
                    report,
                ),
            )
