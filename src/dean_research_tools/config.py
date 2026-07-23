"""Configuration for the researcher tools package."""

from __future__ import annotations

import os

from pydantic import BaseModel, ConfigDict, SecretStr


class Settings(BaseModel):
    """Environment-backed settings for DB and Azure OpenAI."""

    model_config = ConfigDict(frozen=True)

    db_host: str
    db_port: int
    db_name: str
    db_user: str
    db_password: SecretStr
    azure_api_key: SecretStr
    azure_endpoint: str
    embedding_endpoint: str
    embedding_model: str

    @property
    def db_dsn(self) -> SecretStr:
        return SecretStr(
            f"host={self.db_host} "
            f"port={self.db_port} "
            f"dbname={self.db_name} "
            f"user={self.db_user} "
            f"password={self.db_password.get_secret_value()} "
            f"connect_timeout=300 "
            f"keepalives=1 "
            f"keepalives_idle=30 "
            f"keepalives_interval=10 "
            f"keepalives_count=5"
        )


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _research_env(name: str) -> str:
    return f"RESEARCH_{name.upper()}"


def load_settings(
    *,
    db_host: str | None = None,
    db_port: int | None = None,
    db_name: str | None = None,
    db_user: str | None = None,
    db_password: str | SecretStr | None = None,
    azure_api_key: str | SecretStr | None = None,
    azure_endpoint: str | None = None,
    embedding_endpoint: str | None = None,
    embedding_model: str | None = None,
) -> Settings:
    """Load settings from keyword overrides or RESEARCH_* environment variables."""
    resolved_azure_endpoint = azure_endpoint or _require_env(
        _research_env("azure_endpoint")
    )
    resolved_db_host = db_host or _require_env(_research_env("db_host"))
    resolved_db_port = db_port or int(os.getenv(_research_env("db_port"), "5432"))
    resolved_db_name = db_name or _require_env(_research_env("db_name"))
    resolved_db_user = db_user or _require_env(_research_env("db_user"))
    if db_password is None:
        resolved_db_password = SecretStr(_require_env(_research_env("db_password")))
    elif isinstance(db_password, SecretStr):
        resolved_db_password = db_password
    else:
        resolved_db_password = SecretStr(db_password)
    resolved_embedding_endpoint = embedding_endpoint or _require_env(
        _research_env("embedding_endpoint")
    )
    resolved_embedding_deployment = embedding_model or _require_env(
        _research_env("embedding_model")
    )

    if azure_api_key is None:
        resolved_api_key = SecretStr(_require_env(_research_env("azure_api_key")))
    elif isinstance(azure_api_key, SecretStr):
        resolved_api_key = azure_api_key
    else:
        resolved_api_key = SecretStr(azure_api_key)

    return Settings(
        db_host=resolved_db_host,
        db_port=resolved_db_port,
        db_name=resolved_db_name,
        db_user=resolved_db_user,
        db_password=resolved_db_password,
        azure_api_key=resolved_api_key,
        azure_endpoint=resolved_azure_endpoint,
        embedding_endpoint=resolved_embedding_endpoint,
        embedding_model=resolved_embedding_deployment,
    )
