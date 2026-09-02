import logging

import pytest
from pydantic import ValidationError

import ticket_pipeline.language as language_module
from ticket_pipeline.language import (
    LanguageDetectionConfig,
    LanguageDetector,
    LanguagePrediction,
    annotate_ticket_language,
    build_language_text,
)
from ticket_pipeline.models import (
    LanguageDetectionMetadata,
    LanguageDetectionStatus,
    Ticket,
)


class StubLanguageIdentifier:
    def __init__(
        self,
        prediction: LanguagePrediction | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.prediction = prediction
        self.error = error
        self.calls = 0
        self.texts: list[str] = []

    def predict(self, text: str) -> LanguagePrediction:
        self.calls += 1
        self.texts.append(text)
        if self.error is not None:
            raise self.error
        assert self.prediction is not None
        return self.prediction


def make_ticket(**updates: object) -> Ticket:
    values: dict[str, object] = {
        "subject": "Login problem",
        "message": "I cannot log into my account.",
        "ticket_type": "Incident",
        "queue": "Technical Support",
        "priority": "high",
        "language": "en",
        "source_version": 52,
        "tags": ["login"],
    }
    values.update(updates)
    return Ticket.model_validate(values)


@pytest.fixture(scope="module")
def real_detector() -> LanguageDetector:
    return LanguageDetector()


@pytest.mark.parametrize(
    ("text", "expected_language"),
    [
        (
            "I am writing to report that I cannot log into my account "
            "after the recent update.",
            "en",
        ),
        (
            "Estoy escribiendo para informar que no puedo iniciar sesión "
            "en mi cuenta después de la actualización reciente.",
            "es",
        ),
        (
            "Ich schreibe, um zu melden, dass ich mich nach dem letzten "
            "Update nicht mehr in mein Konto einloggen kann.",
            "de",
        ),
    ],
)
def test_real_detector_identifies_clear_languages(
    real_detector: LanguageDetector,
    text: str,
    expected_language: str,
) -> None:
    result = real_detector.detect(text)

    assert result.language == expected_language
    assert result.status is LanguageDetectionStatus.DETECTED
    assert result.confidence is not None
    assert 0.8 <= result.confidence <= 1.0


@pytest.mark.parametrize("text", ["OK", "", "   ", None])
def test_short_or_empty_input_is_uncertain_without_backend_call(
    text: str | None,
) -> None:
    backend = StubLanguageIdentifier(LanguagePrediction("en", 0.99))
    detector = LanguageDetector(backend=backend)

    result = detector.detect(text)

    assert result == LanguageDetectionMetadata(
        language=None,
        confidence=None,
        status=LanguageDetectionStatus.UNCERTAIN,
    )
    assert backend.calls == 0


@pytest.mark.parametrize(
    ("confidence", "expected_status", "expected_language"),
    [
        (0.79, LanguageDetectionStatus.UNCERTAIN, None),
        (0.80, LanguageDetectionStatus.DETECTED, "en"),
        (0.95, LanguageDetectionStatus.DETECTED, "en"),
    ],
)
def test_confidence_policy_including_exact_boundary(
    confidence: float,
    expected_status: LanguageDetectionStatus,
    expected_language: str | None,
) -> None:
    backend = StubLanguageIdentifier(LanguagePrediction("en", confidence))
    detector = LanguageDetector(backend=backend)

    result = detector.detect("Payment failed during checkout")

    assert result.status is expected_status
    assert result.language == expected_language
    assert result.confidence == confidence


def test_threshold_and_minimum_text_are_configurable() -> None:
    backend = StubLanguageIdentifier(LanguagePrediction("en", 0.61))
    detector = LanguageDetector(
        LanguageDetectionConfig(
            confidence_threshold=0.60,
            min_alphabetic_characters=2,
        ),
        backend=backend,
    )

    result = detector.detect("Hi")

    assert result.status is LanguageDetectionStatus.DETECTED


def test_one_backend_instance_is_reused_across_records() -> None:
    backend = StubLanguageIdentifier(LanguagePrediction("en", 0.95))
    detector = LanguageDetector(backend=backend)

    first = detector.detect("Payment failed during checkout")
    second = detector.detect("Account access stopped working")

    assert first.status is LanguageDetectionStatus.DETECTED
    assert second.status is LanguageDetectionStatus.DETECTED
    assert backend.calls == 2


def test_initialization_failure_returns_failed_and_is_logged(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fail_initialization() -> StubLanguageIdentifier:
        raise RuntimeError("model initialization failed")

    monkeypatch.setattr(
        language_module,
        "LinguaLanguageIdentifier",
        fail_initialization,
    )

    with caplog.at_level(logging.ERROR):
        detector = LanguageDetector()
        result = detector.detect("Payment failed during checkout")

    assert result.status is LanguageDetectionStatus.FAILED
    assert "Language detector initialization failed" in caplog.text


@pytest.mark.parametrize(
    "config",
    [
        {"confidence_threshold": -0.1},
        {"confidence_threshold": 1.1},
        {"confidence_threshold": True},
        {"min_alphabetic_characters": 0},
        {"min_alphabetic_characters": True},
    ],
)
def test_invalid_configuration_is_rejected(config: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        LanguageDetectionConfig(**config)  # type: ignore[arg-type]


def test_backend_exception_returns_failed_and_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    detector = LanguageDetector(
        backend=StubLanguageIdentifier(error=RuntimeError("backend unavailable"))
    )

    with caplog.at_level(logging.ERROR):
        result = detector.detect("Payment failed during checkout")

    assert result.status is LanguageDetectionStatus.FAILED
    assert result.language is None
    assert result.confidence is None
    assert "Language detection execution failed" in caplog.text
    assert "Payment failed during checkout" not in caplog.text


@pytest.mark.parametrize(
    "prediction",
    [
        LanguagePrediction("English", 0.99),
        LanguagePrediction("EN", 0.99),
        LanguagePrediction("en", -0.1),
        LanguagePrediction("en", 1.1),
        LanguagePrediction("en", float("nan")),
    ],
)
def test_invalid_backend_prediction_fails_safely(
    prediction: LanguagePrediction,
    caplog: pytest.LogCaptureFixture,
) -> None:
    detector = LanguageDetector(backend=StubLanguageIdentifier(prediction))

    with caplog.at_level(logging.ERROR):
        result = detector.detect("Payment failed during checkout")

    assert result.status is LanguageDetectionStatus.FAILED
    assert "invalid prediction" in caplog.text


def test_language_metadata_validation_enforces_contract() -> None:
    with pytest.raises(ValidationError):
        LanguageDetectionMetadata(
            language=None,
            confidence=0.95,
            status=LanguageDetectionStatus.DETECTED,
        )
    with pytest.raises(ValidationError):
        LanguageDetectionMetadata(
            language="English",
            confidence=0.95,
            status=LanguageDetectionStatus.DETECTED,
        )


def test_annotation_preserves_text_source_metadata_and_non_english_ticket() -> None:
    ticket = make_ticket(
        subject="Problema de acceso",
        message="No puedo iniciar sesión en mi cuenta.",
        language="es",
    )
    backend = StubLanguageIdentifier(LanguagePrediction("es", 0.96))
    detector = LanguageDetector(backend=backend)

    annotated = annotate_ticket_language(ticket, detector)

    assert annotated is not ticket
    assert annotated.subject == ticket.subject
    assert annotated.message == ticket.message
    assert annotated.ticket_type == ticket.ticket_type
    assert annotated.language == "es"
    assert annotated.language_detection is not None
    assert annotated.language_detection.language == "es"
    assert annotated.language_detection.status is LanguageDetectionStatus.DETECTED
    assert backend.texts == [
        "Problema de acceso\nNo puedo iniciar sesión en mi cuenta."
    ]


def test_build_language_text_uses_only_customer_text() -> None:
    assert build_language_text("Login problem", "Cannot sign in") == (
        "Login problem\nCannot sign in"
    )
    assert build_language_text(None, "Cannot sign in") == "Cannot sign in"


def test_repeated_detection_is_deterministic(real_detector: LanguageDetector) -> None:
    text = "No puedo iniciar sesión en mi cuenta."

    first = real_detector.detect(text)
    second = real_detector.detect(text)

    assert first.language == second.language
    assert first.status == second.status
    # Lingua's own float scoring can differ in its last significant digit
    # between calls on a shared, already-warmed detector; repeated detection
    # is deterministic in outcome, not necessarily bit-identical in confidence.
    assert first.confidence == pytest.approx(second.confidence, rel=1e-9)
