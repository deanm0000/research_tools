"""Embedding model construction for retrieval queries."""

from __future__ import annotations

from typing import Literal

from cohere import AsyncClientV2
from pydantic import BaseModel, SecretStr

from dean_research_tools.config import SettingsLike, load_settings


class EmbeddingsModel:
    def __init__(
        self,
        *,
        settings: SettingsLike | None = None,
        embedding_model: str | None = None,
        api_key: str | None = None,
        embedding_endpoint: str | None = None,
    ):
        if settings is None and (
            embedding_model is None or api_key is None or embedding_endpoint is None
        ):
            settings = load_settings()
        if settings is not None and (
            embedding_model is not None
            or api_key is not None
            or embedding_endpoint is not None
        ):
            raise ValueError(
                "If settings is provided, embedding_model, api_key, and embedding_endpoint must all be None"
            )
        if settings is not None:
            if isinstance(settings, BaseModel):
                settings = settings.model_dump()
            embedding_model = settings.get("embedding_model")
            api_key = settings.get("azure_api_key")
            if isinstance(api_key, SecretStr):
                api_key = api_key.get_secret_value()
            embedding_endpoint = settings.get("embedding_endpoint")
        if embedding_model is None or api_key is None or embedding_endpoint is None:
            raise ValueError(
                "Must provide either settings or all of embedding_model, api_key, and embedding_endpoint"
            )
        self.embedding_model = embedding_model

        self.co_client = AsyncClientV2(
            api_key=api_key,
            base_url=embedding_endpoint,
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

    async def embed_search(self, text: str):
        result = await self.co_client.embed(
            model=self.embedding_model,
            texts=[text],
            input_type="search_query",
            embedding_types=["float"],
        )
        flt = result.embeddings.float_
        if flt is None:
            raise ValueError("Embedding result is None")
        return flt[0]

    @staticmethod
    def chunk_list(lst, size=96):
        return [lst[i : i + size] for i in range(0, len(lst), size)]
