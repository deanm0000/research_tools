import os
from pathlib import Path

import pytest
from psycopg import AsyncConnection

from dean_research_tools import PGTools
from dean_research_tools.config import load_settings
from dean_research_tools.embeddings import EmbeddingsModel


def is_local():
    env_path = Path(".env")
    if not env_path.exists():
        return False

    required = {
        "RESEARCH_AZURE_API_KEY",
        "RESEARCH_AZURE_ENDPOINT",
        "RESEARCH_DB_HOST",
        "RESEARCH_DB_NAME",
        "RESEARCH_DB_PASSWORD",
        "RESEARCH_DB_USER",
        "RESEARCH_EMBEDDING_ENDPOINT",
        "RESEARCH_EMBEDDING_MODEL",
        "RESEARCH_QUEUE_CONN_STR",
    }
    configured = set()
    for line in env_path.read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator and value.strip():
            configured.add(key.strip())
    return required <= configured


def read_env_file(to_env: bool = False):
    with Path(".env").open("r") as f:
        env_file = f.readlines()
    settings_dict = {}
    for line in env_file:
        k, v = line.split("=", maxsplit=1)
        if not k.startswith("RESEARCH_"):
            continue
        settings_k = k[9:].lower()

        v = v.strip()
        if v[0] == '"' and v[-1] == '"':
            v = v[1:-1]
        settings_dict[settings_k] = v
        if to_env:
            os.environ[k] = v
    if to_env:
        return
    return load_settings(**settings_dict)


@pytest.mark.parametrize("with_env", [True, False])
@pytest.mark.asyncio
async def test_embed(with_env: bool):
    if not is_local():
        pytest.skip(".env does not contain all required RESEARCH_* settings")
    if with_env:
        read_env_file(to_env=True)
        settings = None
    else:
        settings = read_env_file(to_env=False)
    emb = EmbeddingsModel(settings=settings)
    await emb.embed_texts(["hello"])


@pytest.mark.parametrize("with_env", [True, False])
@pytest.mark.asyncio
async def test_semantic_search(with_env: bool):
    if not is_local():
        pytest.skip(".env does not contain all required RESEARCH_* settings")
    if with_env:
        read_env_file(to_env=True)
        settings = None
    else:
        settings = read_env_file(to_env=False)
    tools = PGTools(settings=settings)

    await tools.semantic_browser_task_search("poop")


@pytest.mark.asyncio
async def test_pool():
    if not is_local():
        pytest.skip(".env does not contain all required RESEARCH_* settings")
    from psycopg_pool import AsyncConnectionPool

    settings = read_env_file()
    assert settings is not None
    async with AsyncConnectionPool(
        conninfo=settings.db_dsn.get_secret_value(),
        connection_class=AsyncConnection,
    ) as pool:
        assert pool is not None
        pgtools = PGTools(conn_or_pool=pool)
        async with pgtools._get_curt() as cur:
            await cur.execute("SELECT 1")
            result = await cur.fetchone()
            assert result is not None
            assert result[0] == 1


@pytest.mark.parametrize("with_env", [True, False])
@pytest.mark.asyncio
async def test_semantic_research_search(with_env: bool):
    if not is_local():
        pytest.skip(".env does not contain all required RESEARCH_* settings")
    if with_env:
        read_env_file(to_env=True)
        settings = None
    else:
        settings = read_env_file(to_env=False)
    tools = PGTools(settings=settings)

    await tools.semantic_research_search("san diego ppa")
