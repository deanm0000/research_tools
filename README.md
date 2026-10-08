# dean_research_tools

I made this library so I could use it in different environments without copying and pasting. I'm not intending to have it be used by a lot of people.

If a lot of people want to use it then that's fine, there's no secret sauce, but you're on your own.

## Browser database

Browser queries use the `ai_proj_browser` schema. Content searches read
`text_content`, join `docs` and `tasks` by `doc_id` and `task_id`, and return
`chunk_indx` as `chunk_index`. Image content is not included in these text searches.

`PGTools.queue_browser` requires a keyword-only `deployment_id` referencing
`ai_proj.models`. Pass it as
`await tools.queue_browser(research_task_id, state, browser_task, vector, deployment_id=deployment_id)`.
The task insert stores `research_task_id` but does not store `state`, which has no
column in the browser task table.