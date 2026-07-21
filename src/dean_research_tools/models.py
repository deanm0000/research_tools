from pydantic import BaseModel, Field


class SemanticContentSearchInput(BaseModel):
    query: str = Field(
        description="The search query string to be embedded with cohere and used for vector similarity search on text content."
    )
    task_id: int | None = Field(
        default=None,
        description=(
            "Optional task scope. Omit this for corpus-wide searches. "
            "Use it when the question is restricted to a browser task."
        ),
    )
    url_part: str | None = Field(
        default=None,
        description=(
            "Optional URL scope. Omit this for corpus-wide searches. "
            "It will filter for content whose URL contains this substring."
        ),
    )
    doc_id: int | None = Field(
        default=None,
        description="Optional doc_id to filter the search results. If provided, only content from this document will be considered.",
    )
    top_k: int = Field(
        default=5,
        description="The number of top results to return (default is 5).",
    )
    min_score: float = Field(
        default=0,
        description="Minimum score threshold for filtering results (between 0.0 and 1.0)",
    )


class SemanticTaskSearchInput(BaseModel):
    query: str = Field(
        description="The search query string to be embedded with cohere and used for vector similarity search on previous browser tasks."
    )
    top_k: int = Field(
        default=5,
        description="The number of top results to return (default is 5).",
    )
    min_score: float = Field(
        default=0,
        description="Minimum score threshold for filtering results (between 0.0 and 1.0)",
    )


class KeywordContentSearchInput(BaseModel):
    keywords: list[str] = Field(
        description="The keywords to search for in the text_chunk of the browser_content table. The search will be case-insensitive and will match any text_chunk that contains any of the keywords."
    )
    task_id: int | None = Field(
        default=None,
        description="The task_id by which to filter the search results. If this isn't provided then url_part must be provided.",
    )
    url_part: str | None = Field(
        default=None,
        description=(
            "Optional URL scope. Omit this for corpus-wide searches. "
            "It will filter for content whose URL contains this substring."
        ),
    )
    doc_id: int | None = Field(
        default=None,
        description="Optional doc_id to further filter the search results.",
    )
    pages: list[int] | None = Field(
        default=None,
        description="Optional list of page numbers to filter the search results. Can only be provided if doc_id is also provided.",
    )
    top_k: int = Field(
        default=5,
        description="The number of top results to return (default is 5).",
    )


class GetTaskInput(BaseModel):
    task_id: int = Field(
        description="The task_id from which to retrieve the associated task details."
    )


class BrowserUseInput(BaseModel):
    objective: str = Field(
        description="""Invoke an agent using the browser_use library to use a real browser to collect content. The agent needs to be given a detailed objective of what content
        it needs to ingest. The agent will then use the browser to collect content and ingest it into the database. The agent will return a task_id which can be used
        to retrieve the content with your existing tools."""
    )
