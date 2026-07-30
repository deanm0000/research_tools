"""Embedding model construction for retrieval queries."""

from __future__ import annotations

from typing import Literal

from cohere import AsyncClientV2

from dean_research_tools.config import SettingsLike, load_settings


class EmbeddingsModel:
    def __init__(
        self,
        *,
        settings: SettingsLike | None = None,
    ):
        settings = load_settings(settings)

        self.embedding_model = settings.embedding_model

        self.co_client = AsyncClientV2(
            api_key=settings.azure_api_key.get_secret_value(),
            base_url=settings.embedding_endpoint,
        )

    async def embed_texts(
        self,
        texts: list[str],
        input_type: Literal["search_document", "search_query"] = "search_query",
    ) -> list[list[float]] | None:
        results = []
        chunked = self.chunk_list(texts)
        for chunk in chunked:
            result = await self.co_client.embed(
                model=self.embedding_model,
                texts=chunk,
                input_type=input_type,
                embedding_types=["float"],
            )
            flt = result.embeddings.float_
            if flt is None:
                return None
            results.extend(flt)
        return results

    @staticmethod
    def chunk_list(lst, size=96):
        return [lst[i : i + size] for i in range(0, len(lst), size)]
