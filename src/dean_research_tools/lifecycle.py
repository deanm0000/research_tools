"""Trusted lifecycle operations with caller-owned connections and transactions."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Literal, TypeAlias

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


class BrowserRequest(BaseModel):
    objective: str
    embedding: list[float]
    ordinal: int
    tool_call_id: str | None = None


class BrowserContinuation(BaseModel):
    batch_id: int
    research_id: int


class ResearchContinuation(BrowserContinuation):
    """A finished subresearch task was elected to resume its parent's batch."""


ResearchRole: TypeAlias = Literal["root", "claim_check", "subtask"]
TerminalResearchStatus: TypeAlias = Literal["complete", "incomplete"]


class SubresearchRequest(BaseModel):
    question: str
    embedding: list[float]
    ordinal: int
    tool_call_id: str | None = None
    role: Literal["claim_check", "subtask"] = "subtask"
    context: str | None = None
    standalone: bool = False


class BatchMember(BaseModel):
    kind: Literal["browser", "subresearch"]
    task_id: int
    ordinal: int


class PendingMember(BaseModel):
    kind: Literal["browser", "subresearch"]
    task_id: int
    status: str
    finished: bool


class ResearchFinish(BaseModel):
    final_status: TerminalResearchStatus
    continuation: ResearchContinuation | None = None


class ResearchNode(BaseModel):
    id: int
    starting_task: str
    status: str
    parent_id: int | None
    root_id: int
    depth: int
    role: ResearchRole
    context: str | None
    standalone: bool
    triage: dict[str, JsonValue] | None
    user_id: str


class ResearchChild(BaseModel):
    id: int
    starting_task: str
    status: str
    role: ResearchRole
    standalone: bool
    title: str | None


class TreeCost(BaseModel):
    root_id: int
    cost: Decimal
    cost_cap_usd: Decimal
    unpriced_lines: int

    @property
    def over_cap(self) -> bool:
        return self.cost >= self.cost_cap_usd


class TreeLimits(BaseModel):
    root_id: int
    depth: int
    max_depth: int
    max_batch_children: int
    tree_children: int
    max_tree_children: int

    @property
    def can_spawn(self) -> bool:
        return self.depth < self.max_depth and self.tree_children < self.max_tree_children

    def spawn_allowance(self) -> int:
        """How many children one new batch may still create."""
        if self.depth >= self.max_depth:
            return 0
        return max(0, min(self.max_batch_children, self.max_tree_children - self.tree_children))


class ResearchUsage(BaseModel):
    """One model call's tokens; input_tokens excludes cached_input_tokens."""

    deployment_name: str
    flex: bool = False
    input_tokens: int
    cached_input_tokens: int = 0
    output_tokens: int
    purpose: str = "agent"


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

    async def get_pending_browsers(self, research_id: int) -> list[BrowserDispatch]:
        async with self.conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """SELECT t.task_id, t.status::text AS status,
                    (d.task_id IS NOT NULL) AS finished
                FROM ai_proj.research_task_pause p
                JOIN ai_proj.research_browser_members m USING (batch_id)
                JOIN ai_proj_browser.tasks t ON t.task_id=m.task_id
                LEFT JOIN ai_proj_browser.tasks_done d ON d.task_id=t.task_id
                WHERE p.research_id=%s ORDER BY m.ordinal""",
                (research_id,),
            )
            rows = await cur.fetchall()
        return [BrowserDispatch.model_validate(row) for row in rows]

    async def create_browser_batch(
        self, research_id: int, state: State, requests: list[BrowserRequest],
    ) -> list[int]:
        if len(requests) == 0:
            raise ValueError("A browser batch must contain at least one request")
        async with self.conn.cursor() as cur:
            await cur.execute(
                """SELECT task_id FROM ai_proj.research_make_browser_batch(
                    %s::bigint,%s::jsonb,%s::jsonb)""",
                (research_id, to_json_str(state),
                 TypeAdapter(list[BrowserRequest]).dump_json(requests).decode()),
            )
            rows = await cur.fetchall()
        return [row[0] for row in rows]

    async def get_browser_continuation(
        self, browser_id: int,
    ) -> BrowserContinuation | None:
        async with self.conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """SELECT batch_id, research_id FROM ai_proj.research_browser_batches
                WHERE publisher_task_id=%s AND published_at IS NULL
                  AND resumed_at IS NULL""",
                (browser_id,),
            )
            row = await cur.fetchone()
        return None if row is None else BrowserContinuation.model_validate(row)

    async def acknowledge_browser_continuation(self, batch_id: int) -> None:
        async with self.conn.cursor() as cur:
            await cur.execute(
                """UPDATE ai_proj.research_browser_batches SET published_at=now()
                WHERE batch_id=%s AND published_at IS NULL
                  AND (publisher_task_id IS NOT NULL OR publisher_research_id IS NOT NULL)""",
                (batch_id,),
            )

    async def accepts_continuation(self, research_id: int, batch_id: int) -> bool:
        async with self.conn.cursor() as cur:
            await cur.execute(
                """SELECT EXISTS (
                    SELECT 1 FROM ai_proj.research_browser_batches b
                    JOIN ai_proj.researcher_tasks r ON r.id=b.research_id
                    WHERE b.batch_id=%s AND b.research_id=%s
                      AND (b.publisher_task_id IS NOT NULL
                           OR b.publisher_research_id IS NOT NULL)
                      AND r.status IN ('paused_for_browser','paused_for_subresearch',
                                       'error','in_progress')
                      AND NOT EXISTS (
                        SELECT 1 FROM ai_proj.research_browser_batches newer
                        WHERE newer.research_id=b.research_id
                          AND newer.batch_id>b.batch_id))""",
                (batch_id, research_id),
            )
            row = await cur.fetchone()
        if row is None:
            raise RuntimeError(f"Could not inspect continuation {batch_id}")
        return row[0]

    async def get_pending_browser(self, research_id: int) -> BrowserDispatch | None:
        """Singleton compatibility: prefer unfinished work when a batch is pending."""
        tasks = await self.get_pending_browsers(research_id)
        if len(tasks) == 0:
            return None
        return next((task for task in tasks if not task.finished), tasks[0])

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
        """Check for a committed terminal status (complete, or incomplete at the cost cap).

        An incomplete task keeps its done row when reopened, so status is authoritative.
        """
        async with self.conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1 FROM ai_proj.researcher_tasks
                    WHERE id = %s AND status IN ('complete', 'incomplete')
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

    async def start_research(
        self, research_task_id: int, *, recover_in_progress: bool = False,
    ) -> ResearchStart:
        """Recover an in-progress checkpoint only while holding an exclusive worker claim."""
        async with self.conn.cursor(row_factory=dict_row) as cur:
            if recover_in_progress:
                await cur.execute(
                    "SELECT * FROM ai_proj.start_research_task(%s::integer,%s::boolean)",
                    (research_task_id, True),
                )
            else:
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
                     WHERE EXISTS (
                         SELECT 1 FROM ai_proj_browser.tasks_docs scope
                         JOIN ai_proj_browser.docs d USING (doc_id)
                         WHERE scope.task_id = t.task_id
                           AND d.blake3 = c.blake3
                     )) AS text_chunks,
                    (SELECT count(*) FROM ai_proj_browser.image_content c
                     WHERE EXISTS (
                         SELECT 1 FROM ai_proj_browser.tasks_docs scope
                         JOIN ai_proj_browser.docs d USING (doc_id)
                         WHERE scope.task_id = t.task_id
                           AND d.blake3 = c.blake3
                     )) AS image_chunks,
                    ARRAY(
                        SELECT b.research_id FROM ai_proj.research_browser_batches b
                        JOIN ai_proj.research_task_pause p USING (batch_id)
                        WHERE b.publisher_task_id=t.task_id
                          AND b.published_at IS NULL AND b.resumed_at IS NULL
                        ORDER BY b.research_id
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
        terminal_status: TerminalResearchStatus = "complete",
    ) -> ResearchFinish:
        """Save completion and an optional report atomically, without LLM calls.

        The returned continuation is set when this task was the last unfinished member of its
        parent's batch; the caller must then resume the parent.
        """
        completed_state = state if usage is None else {**state, "usage": usage}
        async with self.conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                WITH finished AS MATERIALIZED (
                    SELECT %s::bigint AS id, f.final_status, f.parent_research_id, f.batch_id
                    FROM ai_proj.finish_research_task(
                        %s::bigint, %s::jsonb, %s::text, %s::text
                    ) f
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
                SELECT f.final_status, f.parent_research_id, f.batch_id,
                    (SELECT count(*) FROM report_saved) AS reports_saved
                FROM finished f
                """,
                (
                    research_task_id,
                    research_task_id,
                    to_json_str(completed_state),
                    summary,
                    terminal_status,
                    report,
                    report,
                ),
            )
            row = await cur.fetchone()
        if row is None:
            raise RuntimeError(f"Research task {research_task_id} was not finished")
        continuation = None
        if row["parent_research_id"] is not None:
            continuation = ResearchContinuation(
                batch_id=row["batch_id"], research_id=row["parent_research_id"],
            )
        return ResearchFinish(final_status=row["final_status"], continuation=continuation)

    async def create_batch(
        self,
        research_id: int,
        state: State,
        browsers: list[BrowserRequest],
        children: list[SubresearchRequest],
        triage: dict[str, JsonValue] | None = None,
    ) -> list[BatchMember]:
        """Pause research on browser and/or subresearch members; SQL enforces tree caps."""
        if len(browsers) + len(children) == 0:
            raise ValueError("A batch must contain at least one request")
        async with self.conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """SELECT kind, task_id, ordinal FROM ai_proj.research_make_batch(
                    %s::bigint,%s::jsonb,%s::jsonb,%s::jsonb,%s::jsonb)""",
                (
                    research_id,
                    to_json_str(state),
                    TypeAdapter(list[BrowserRequest]).dump_json(browsers).decode(),
                    TypeAdapter(list[SubresearchRequest]).dump_json(children).decode(),
                    None if triage is None else to_json_str(triage),
                ),
            )
            rows = await cur.fetchall()
        return sorted(
            (BatchMember.model_validate(row) for row in rows), key=lambda m: m.ordinal,
        )

    async def get_pending_members(self, research_id: int) -> list[PendingMember]:
        """Members of the batch a paused research task is waiting on, in ordinal order."""
        async with self.conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """SELECT kind, task_id, status, finished FROM (
                    SELECT m.ordinal, 'browser' AS kind, t.task_id, t.status::text AS status,
                        (d.task_id IS NOT NULL) AS finished
                    FROM ai_proj.research_task_pause p
                    JOIN ai_proj.research_browser_members m USING (batch_id)
                    JOIN ai_proj_browser.tasks t ON t.task_id=m.task_id
                    LEFT JOIN ai_proj_browser.tasks_done d ON d.task_id=t.task_id
                    WHERE p.research_id=%s
                    UNION ALL
                    SELECT c.ordinal, 'subresearch', r.id, r.status::text,
                        r.status IN ('complete','incomplete')
                    FROM ai_proj.research_task_pause p
                    JOIN ai_proj.research_child_members c USING (batch_id)
                    JOIN ai_proj.researcher_tasks r ON r.id=c.child_id
                    WHERE p.research_id=%s
                ) members ORDER BY ordinal""",
                (research_id, research_id),
            )
            rows = await cur.fetchall()
        return [PendingMember.model_validate(row) for row in rows]

    async def get_research_continuation(
        self, research_id: int,
    ) -> ResearchContinuation | None:
        """The unpublished parent batch this finished subresearch task was elected to resume."""
        async with self.conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """SELECT batch_id, research_id FROM ai_proj.research_browser_batches
                WHERE publisher_research_id=%s AND published_at IS NULL
                  AND resumed_at IS NULL
                ORDER BY batch_id DESC LIMIT 1""",
                (research_id,),
            )
            row = await cur.fetchone()
        return None if row is None else ResearchContinuation.model_validate(row)

    async def get_research_node(self, research_id: int) -> ResearchNode:
        async with self.conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """SELECT id, starting_task, status::text AS status, parent_id, root_id,
                    depth, role, context, standalone, triage, user_id
                FROM ai_proj.researcher_tasks WHERE id=%s""",
                (research_id,),
            )
            row = await cur.fetchone()
        if row is None:
            raise LookupError(f"Research task {research_id} does not exist")
        return ResearchNode.model_validate(row)

    async def get_children(self, research_id: int) -> list[ResearchChild]:
        """Direct subresearch tasks in creation order, with their saved titles."""
        async with self.conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """SELECT r.id, r.starting_task, r.status::text AS status, r.role,
                    r.standalone, d.summary AS title
                FROM ai_proj.researcher_tasks r
                LEFT JOIN ai_proj.researcher_tasks_done d ON d.id=r.id
                WHERE r.parent_id=%s ORDER BY r.id""",
                (research_id,),
            )
            rows = await cur.fetchall()
        return [ResearchChild.model_validate(row) for row in rows]

    async def tree_cost(self, research_id: int) -> TreeCost:
        """Spend of the whole tree containing this task, priced from model_costs."""
        async with self.conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT * FROM ai_proj.research_tree_cost(%s::bigint)", (research_id,),
            )
            row = await cur.fetchone()
        if row is None:
            raise LookupError(f"Research task {research_id} does not exist")
        return TreeCost.model_validate(row)

    async def tree_limits(self, research_id: int) -> TreeLimits:
        async with self.conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT * FROM ai_proj.research_tree_limits(%s::bigint)", (research_id,),
            )
            row = await cur.fetchone()
        if row is None:
            raise LookupError(f"Research task {research_id} does not exist")
        return TreeLimits.model_validate(row)

    async def record_research_usage(
        self, research_id: int, usage: list[ResearchUsage],
    ) -> None:
        if len(usage) == 0:
            return
        async with self.conn.cursor() as cur:
            await cur.execute(
                "SELECT ai_proj.record_research_usage(%s::bigint,%s::jsonb)",
                (research_id, TypeAdapter(list[ResearchUsage]).dump_json(usage).decode()),
            )

    async def save_research_triage(
        self, research_id: int, triage: dict[str, JsonValue],
    ) -> None:
        async with self.conn.cursor() as cur:
            await cur.execute(
                "SELECT ai_proj.save_research_triage(%s::bigint,%s::jsonb)",
                (research_id, to_json_str(triage)),
            )

    async def reopen_incomplete_tree(self, root_id: int) -> list[int]:
        """Reopen a capped tree after its cap was raised; returns task ids to dispatch."""
        async with self.conn.cursor() as cur:
            await cur.execute(
                "SELECT research_id FROM ai_proj.reopen_incomplete_tree(%s::bigint)",
                (root_id,),
            )
            rows = await cur.fetchall()
        return [row[0] for row in rows]
