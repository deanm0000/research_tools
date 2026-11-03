# dean_research_tools

I made this library so I could use it in different environments without copying and pasting. I'm not intending to have it be used by a lot of people.

If a lot of people want to use it then that's fine, there's no secret sauce, but you're on your own.

## Browser database

Browser queries use the `ai_proj_browser` schema. Content searches read
`text_content`, join `docs` and `tasks` by `doc_id` and `task_id`, and return
`chunk_indx` as `chunk_index`. Image content is not included in these text searches.
Task-scoped content queries include documents owned by the requested task and
documents reused through its completed (`done`) links. An `EXISTS` filter avoids
duplicating chunks when multiple links reuse a document. Returned `task_id` remains
the document's original owning task, which may differ from the requested scope.

Both semantic content and browser-task searches enforce `min_score` in SQL before
applying `top_k`. Similarity thresholds must be between 0 and 1.

## Trusted lifecycle APIs (0.13.0)

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
start_research(research_task_id) -> ResearchStart
create_browser(research_task_id, state, objective, vector) -> int
start_browser(browser_task_id) -> BrowserStart
browser_progress(browser_task_id) -> BrowserProgress
finish_browser(browser_task_id, history, usage) -> BrowserProgress
get_browser_completion(browser_task_id) -> BrowserCompletion | None
get_pending_browser(research_id) -> BrowserDispatch | None
mark_research_error(research_id) -> None
is_research_complete(research_id) -> bool
finish_research(research_task_id, state, usage=None, report=None, *, summary=None) -> None
```

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
`pending_research_ids` includes paused tasks only when a stored completion exists
and no links remain queued.

The browser worker must remain alive while `progress.queued > 0`, then call
`finish_browser`. SQL validates readiness and normalizes success, partial success,
or failure. The returned progress is read back from stored data, not inferred
from the caller's raw history. `BrowserCompletion` exposes `task_id`, `status`,
`history` (also available as `result`), `usage`, and `finished_at`.
On a retry, check `get_browser_completion` **before** `start_browser`: an existing
completion lets the worker retry research queue publication without rerunning
browser automation or attempting to restart a completed task.

On research retries, `get_pending_browser` checks for an existing paused handoff
before starting research. Its root-exported `BrowserDispatch` has `task_id: int`,
`status: str`, and `finished: bool` (a stored completion exists). This allows
republication after browser queue send failure without creating another task or
resuming research prematurely. `mark_research_error` changes only `in_progress`
research to `error`, leaving paused and completed tasks untouched.
`is_research_complete` checks for stored research completion so duplicate
continuation messages can be acknowledged without starting completed work.

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
    tests/test_lifecycle.py tests/test_embeddings.py tests/test_queue_settings.py
```

`tests/test_tools_locally.py` contains credential-dependent live integration
checks; it is deliberately not included in the offline command.