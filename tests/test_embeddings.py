from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dean_research_tools.embeddings import EmbeddingsModel


@pytest.fixture
def embeddings() -> EmbeddingsModel:
    model = EmbeddingsModel(
        embedding_model="cohere", api_key="offline", embedding_endpoint="https://invalid.local"
    )
    model.co_client = SimpleNamespace(
        embed=AsyncMock(return_value=SimpleNamespace(
            embeddings=SimpleNamespace(float_=[[0.1, 0.2]])
        ))
    )
    return model


@pytest.mark.asyncio
async def test_query_and_document_purposes(embeddings: EmbeddingsModel) -> None:
    assert await embeddings.embed_search("query") == [0.1, 0.2]
    assert embeddings.co_client.embed.call_args.kwargs["input_type"] == "search_query"
    assert await embeddings.embed_documents(["document"]) == [[0.1, 0.2]]
    assert embeddings.co_client.embed.call_args.kwargs["input_type"] == "search_document"


@pytest.mark.asyncio
async def test_batch_embedding(embeddings: EmbeddingsModel) -> None:
    embeddings.co_client.embed.side_effect = [
        SimpleNamespace(embeddings=SimpleNamespace(float_=[[0.1]] * 96)),
        SimpleNamespace(embeddings=SimpleNamespace(float_=[[0.2]])),
    ]
    result = await embeddings.embed_documents(["doc"] * 97)
    assert len(result) == 97
    assert result[-1] == [0.2]
    assert embeddings.co_client.embed.await_count == 2


@pytest.mark.asyncio
async def test_empty_embeddings(embeddings: EmbeddingsModel) -> None:
    assert await embeddings.embed_documents([]) == []
    embeddings.co_client.embed.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [None, []])
async def test_invalid_embedding_response(embeddings: EmbeddingsModel, result: list[list[float]] | None) -> None:
    embeddings.co_client.embed.return_value = SimpleNamespace(
        embeddings=SimpleNamespace(float_=result)
    )
    with pytest.raises(ValueError):
        await embeddings.embed_search("query")
