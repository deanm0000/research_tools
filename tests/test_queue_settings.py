import pytest

from dean_research_tools.config import Settings, load_settings


def test_default_queue_name() -> None:
    assert Settings.dummies().queue_name == "allworker"


def test_settings_queue_name() -> None:
    settings = Settings.dummies().model_copy(update={"queue_name": "custom"})
    assert load_settings(settings).queue_name == "custom"


def test_environment_queue_name(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings.dummies().model_dump(exclude={"queue_name"})
    for key, value in settings.items():
        if hasattr(value, "get_secret_value"):
            value = value.get_secret_value()
        monkeypatch.setenv(f"RESEARCH_{key.upper()}", str(value))
    monkeypatch.setenv("RESEARCH_QUEUE_NAME", "from-env")
    assert load_settings().queue_name == "from-env"
    assert load_settings(queue_name="override").queue_name == "override"
