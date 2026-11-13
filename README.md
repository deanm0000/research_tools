# dean_research_tools

I made this library so I could use it in different environments without copying and pasting. I'm not intending to have it be used by a lot of people.

If a lot of people want to use it then that's fine, there's no secret sauce, but you're on your own.

## Browser database

Version 0.17 requires the content-addressed `ai_proj_browser` schema:
`doc_content.blake3` is the binary content key, `docs.blake3` is a nullable
reference to it, and `text_content` / `image_content` reference it instead of
`doc_id`. URL provenance stays in `docs`; `tasks_docs` and links remain doc-ID based.

Content searches join chunks to document provenance through `blake3` and return
`chunk_indx` as `chunk_index`. Image content is not included in these text searches.
Task-scoped searches consider only documents associated through `tasks_docs`,
including documents reused through completed (`done`) links. Each chunk is returned
once even when several associated URLs share its hash. The lowest matching `doc_id`
provides deterministic URL/title/citation provenance; task, URL, and document filters
are applied **before** choosing that source. Returned `task_id` is the requested
scope, or the first task associated with the selected source for unscoped searches.
`doc_id` filters and page filters keep their existing public API.

`Lifecycle.browser_progress` likewise counts each text/image chunk once per task,
regardless of how many associated source documents share that content.

Both semantic content and browser-task searches enforce `min_score` in SQL before
applying `top_k`. Similarity thresholds must be between 0 and 1.

## Trusted lifecycle APIs (0.15.0)

`PGTools.queue_browser(research_task_id, state, objective, vector) -> int` is a
programmatic compatibility API, **not an LLM tool**. Tool discovery iterates only
the explicit `AVAILABLE_TOOLS` registry. The LLM-facing `browser_use` tool still
records an objective for the worker to hand off.

Browser creation selects the authoritative deployment from
`ai_proj.current_func_settings` for `function_name = 'browser'`; callers do not
choose a model. Research-linked creation uses
`ai_proj.research_make_browser_task(bigint, jsonb, text, vector)` to atomically
persist paused LangChain state and create browser work. Standalone creation
inserts into `ai_proj_browser.tasks` and returns `task_id`.

The compatibility API commits creation before publishing
`{"function": "browser", "payload": task_id}`. It rejects a supplied connection
that already has an active transaction. Queue selection uses `Settings.queue_name`
or `RESEARCH_QUEUE_NAME`, defaulting to `allworker`. Database creation and queue
publication are separate operations: a publication failure leaves a committed task
which can be explicitly re-enqueued; do not blindly retry creation.

`Lifecycle(conn)` is the reusable service for trusted application orchestration.
It accepts an asynchronous psycopg connection and never owns its connection,
transaction, commit, or queue dispatch. Configure pgvector adaptation on the
connection before using embedding parameters. When dispatching task IDs yourself,
commit before publishing them.

State serialization accepts dictionaries containing live LangChain `BaseMessage`
instances and already serialized state via Pydantic's `TypeAdapter`; no full
`langchain` dependency is needed. It preserves message fields and types without
reconstructing runnable LangChain messages.

### Service signatures

The service and return models are exported from `dean_research_tools`.

```python
create_research(question, user_id, embedding, system_prompt=None) -> int
start_research(research_task_id, *, recover_in_progress=False) -> ResearchStart
create_browser(research_task_id, state, objective, vector) -> int
create_browser_batch(research_id, state, requests) -> list[int]
start_browser(browser_task_id) -> BrowserStart
browser_progress(browser_task_id) -> BrowserProgress
finish_browser(browser_task_id, history, usage) -> BrowserProgress
get_browser_completion(browser_task_id) -> BrowserCompletion | None
get_pending_browser(research_id) -> BrowserDispatch | None
get_pending_browsers(research_id) -> list[BrowserDispatch]
get_browser_continuation(browser_id) -> BrowserContinuation | None
acknowledge_browser_continuation(batch_id) -> None
accepts_continuation(research_id, batch_id) -> bool
mark_research_error(research_id) -> None
is_research_complete(research_id) -> bool
finish_research(research_task_id, state, usage=None, report=None, *, summary=None,
                terminal_status="complete") -> ResearchFinish
# 0.15: research trees (worker migration 008)
create_batch(research_id, state, browsers, children, triage=None) -> list[BatchMember]
get_pending_members(research_id) -> list[PendingMember]
get_research_continuation(research_id) -> ResearchContinuation | None
get_research_node(research_id) -> ResearchNode
get_children(research_id) -> list[ResearchChild]
tree_cost(research_id) -> TreeCost
tree_limits(research_id) -> TreeLimits
record_research_usage(research_id, usage: list[ResearchUsage]) -> None
save_research_triage(research_id, triage) -> None
reopen_incomplete_tree(root_id) -> list[int]
```

### Research trees (0.15)

Version 0.15 requires the worker repository's migrations `008a_research_status_values.sql`
and `008b_research_tree.sql`.

- `create_batch` pauses research on browser requests and `SubresearchRequest`
  children together. A `SubresearchRequest` has `question`, `embedding`, `ordinal`,
  optional `tool_call_id`, `role` (`claim_check` or `subtask`), `context`, and
  `standalone`. Ordinals are shared across both kinds. SQL enforces the depth,
  per-batch, and per-tree caps, plus the cost cap.
- `finish_research` returns `ResearchFinish(final_status, continuation)`.
  `continuation` is a `ResearchContinuation` (a `BrowserContinuation` subclass) when
  this task was the last unfinished member of its parent's batch; the caller then
  resumes the parent. `terminal_status="incomplete"` marks work stopped at the cost
  cap. A task with an incomplete child is always stored as incomplete.
- `is_research_complete` is now status-based: `complete` or `incomplete`. Reopened
  incomplete tasks keep their done row.
- `acknowledge_browser_continuation` and `accepts_continuation` accept batches
  published by either a browser or a child.
- `TreeCost` (`cost`, `cost_cap_usd`, `unpriced_lines`, `over_cap`) and `TreeLimits`
  (`spawn_allowance()`) let the worker check budgets before calling the model or
  spawning children.

`ResearchStart` exposes `starting_task`, `state` (nullable serialized state),
`sys_prompt`, and `model` (deployment name string). `BrowserStart` exposes
`starting_task` and `model`.

`BrowserProgress` exposes task `status`, normalized history `outcome`, `history`,
`errors`, `ingestion_summary`, `total_links`, `pending_links`, `done_links`,
`errored_links`, `text_chunks`, `image_chunks`, and `pending_research_ids`.
Properties `queued`, `done`, and `errored` alias the link counters.
`links_terminal` means there are no queued links; it does **not** mean browser
finalization has succeeded. `finished` checks the task status (`complete` or
`error`), and `has_content` checks the text/image chunk counters.
`pending_research_ids` now includes only batches for which this browser is the
elected publisher and publication is still pending. Finalization, not a progress
read, elects that publisher after every member and its ingestion work is terminal.
Use the continuation APIs to publish a batch-aware message and acknowledge sending.

The browser worker must remain alive while `progress.queued > 0`, then call
`finish_browser`. SQL validates readiness and normalizes success, partial success,
or failure. The returned progress is read back from stored data, not inferred
from the caller's raw history. `BrowserCompletion` exposes `task_id`, `status`,
`history` (also available as `result`), `usage`, and `finished_at`.
On a retry, check `get_browser_completion` **before** `start_browser`: an existing
completion lets the worker retry research queue publication without rerunning
browser automation or attempting to restart a completed task.

On research retries, `get_pending_browsers` checks every member of an existing paused handoff
before starting research. Its root-exported `BrowserDispatch` has `task_id: int`,
`status: str`, and `finished: bool` (a stored completion exists). This allows
republication after browser queue send failure without creating another task or
resuming research prematurely. `mark_research_error` changes only `in_progress`
research to `error`, leaving paused and completed tasks untouched.
`is_research_complete` checks for stored research completion so duplicate
continuation messages can be acknowledged without starting completed work.
The singular `get_pending_browser` remains available for compatibility and prefers
an unfinished member when a batch is pending.

### Parallel browser collection and lifecycle

Version 0.14 requires the worker repository's migration
`006_parallel_browser_batches.sql`. Stop/drain old workers, apply the migration,
then deploy this package and the updated worker together. The singleton creation
entry points and standalone browsers remain supported; existing pauses are
backfilled into singleton batches.

`browser_use(objective)` retains its single-objective tool input. Independent calls
in one model response are collected in `PGTools.browser_requests`, rather than
overwriting one handoff. `browser_requests_started` lets the worker detect failed
preparation and refuse a partial batch. Each `BrowserRequest` has `objective`,
`embedding`, `ordinal`, and optional `tool_call_id`. Sort by `ordinal` after the
tool round; embedding completions can arrive out of order. The worker associates
model tool-call IDs before persistence. `triggered_browser_use` remains only as
legacy singleton metadata; new workers must use the collection.

`create_browser_batch` atomically persists the saved tool transcript, all browser
tasks, and their ordered membership. It does not enqueue. Commit before publishing
each task ID. Inspect `get_pending_browsers` after a partial publication failure;
never recreate the batch to recover queue sends.

`finish_browser` requires READ COMMITTED isolation and serializes finalizers using
a shared batch-row lock before recording completion and checking siblings. Stored
completions are immutable on retry. Once every browser has a stored result and no
queued ingestion, exactly one durable publication intent is assigned to the final
browser. `get_browser_continuation` returns its `batch_id` and `research_id`; publish
`{"function":"researcher","payload":research_id,"batch_id":batch_id}`, then call
`acknowledge_browser_continuation`. Failed publication must propagate so the
browser's original message retries without repeating navigation.

A send followed by an acknowledgment crash can duplicate delivery; this is
at-least-once, not exactly-once. The worker must hold an exclusive research claim
before checking `accepts_continuation` and running research. Busy claims must
remain retryable to avoid losing a continuation delivered during initial dispatch.
`start_research` restores all member outcomes and retains a durable resume
checkpoint. Set `recover_in_progress=True` only while holding that exclusive claim
to recover from process death; ordinary callers still cannot restart active work.
Missing/failed browser infrastructure remains retryable rather than being silently
classified as a terminal browser outcome.

Creation requires an approved `user_id` and selects the configured researcher
deployment. An explicit `system_prompt` creates and assigns a prompt record;
otherwise the newest prompt (`added DESC, sys_id DESC`) is assigned. Research
starts use the prompt assigned to that task rather than selecting a newer prompt.
Completion calls existing `ai_proj.finish_research_task(id, state, summary)` and,
if `report` is provided, writes its next report revision atomically in the same SQL
statement. Optional `usage` is stored under the completed state's `usage` key.
Title generation and report formatting remain application responsibilities.

## Shared embeddings

`EmbeddingsModel.embed_search(text)` produces a single `search_query` embedding.
`embed_documents(texts)` uses `search_document`. `embed_texts(texts,
input_type="search_query")` preserves the legacy default and accepts either purpose
explicitly. Texts are batched in groups of 96; empty input returns an empty list.
Missing or mismatched embeddings raise `ValueError` rather than producing partial
results. `browser_use` embeds its persisted objective as `search_document`;
semantic browser-task lookup embeds retrieval queries as `search_query`.

## Offline validation

```shell
python -m pytest -q tests/test_browser_queries.py tests/test_func_args_sync.py \
    tests/test_lifecycle.py tests/test_embeddings.py tests/test_queue_settings.py \
    tests/test_browser_batches.py
```

`tests/test_tools_locally.py` contains credential-dependent live integration
checks; it is deliberately not included in the offline command.