import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_allowed_classes_from_a_comma_separated_environment_variable(monkeypatch) -> None:
    monkeypatch.setenv("AI_ALLOWED_CLASSES", " person , cell phone,,laptop ")
    assert Settings(_env_file=None).allowed_classes == ["person", "cell phone", "laptop"]


def test_empty_allowed_classes_means_all(monkeypatch) -> None:
    monkeypatch.setenv("AI_ALLOWED_CLASSES", "")
    assert Settings(_env_file=None).allowed_classes == []


def test_defaults_are_the_initial_detection_classes() -> None:
    assert Settings(_env_file=None).allowed_classes == ["person", "cell phone", "laptop", "book"]


def test_token_is_never_printed(monkeypatch) -> None:
    monkeypatch.setenv("AI_SERVICE_TOKEN", "super-secret")
    settings = Settings(_env_file=None)
    assert "super-secret" not in repr(settings)
    assert settings.service_token is not None
    assert settings.service_token.get_secret_value() == "super-secret"


@pytest.mark.parametrize("field,value", [("confidence_threshold", 1.5), ("image_size", 32)])
def test_rejects_out_of_range_values(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})


def test_blank_token_does_not_count_as_secure() -> None:
    with pytest.raises(RuntimeError):
        Settings(_env_file=None, service_token="").assert_secure()
