"""Local, confidence-aware language detection for processed ticket text."""

import logging
import math
import re
from dataclasses import dataclass
from typing import Protocol

from ticket_pipeline.models import (
    LanguageDetectionMetadata,
    LanguageDetectionStatus,
    Ticket,
)


DEFAULT_CONFIDENCE_THRESHOLD = 0.80
DEFAULT_MIN_ALPHABETIC_CHARACTERS = 4
# Backends such as Lingua compute confidence with multithreaded floating-point
# summation, so the raw value jitters in its least-significant bits between
# otherwise identical runs. Quantizing before the value is stored keeps
# repeated runs over the same input byte-identical (and therefore keeps the
# content-addressed dataset version stable) while staying far more precise
# than any reported statistic.
_CONFIDENCE_DECIMAL_PLACES = 9
_ISO_639_1_PATTERN = re.compile(r"^[a-z]{2}$")
logger = logging.getLogger(__name__)

# A curated set of languages plausible in a global support-ticket dataset.
# Building the Lingua backend from all 75 supported languages spreads its
# per-language probability mass thin and pushes native confidence down even
# for unambiguous text; restricting to a realistic candidate set keeps
# Lingua's own confidence values meaningful without narrowing detection to
# only the languages already observed in the current dataset (en, de).
SUPPORTED_LANGUAGE_NAMES = (
    "ENGLISH",
    "GERMAN",
    "SPANISH",
    "FRENCH",
    "ITALIAN",
    "PORTUGUESE",
    "DUTCH",
    "POLISH",
    "RUSSIAN",
    "TURKISH",
    "SWEDISH",
    "ARABIC",
    "HEBREW",
    "HINDI",
    "CHINESE",
    "JAPANESE",
    "KOREAN",
    "VIETNAMESE",
    "THAI",
    "INDONESIAN",
)


@dataclass(frozen=True, slots=True)
class LanguageDetectionConfig:
    """Small set of configurable language-detection policies."""

    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    min_alphabetic_characters: int = DEFAULT_MIN_ALPHABETIC_CHARACTERS

    def __post_init__(self) -> None:
        threshold = self.confidence_threshold
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)
            or not 0.0 <= threshold <= 1.0
        ):
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        minimum = self.min_alphabetic_characters
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
            raise ValueError("min_alphabetic_characters must be a positive integer")


@dataclass(frozen=True, slots=True)
class LanguagePrediction:
    """Raw prediction returned by an interchangeable detector backend."""

    language: str
    confidence: float


class LanguageIdentifier(Protocol):
    """Minimal interface required from a language-identification backend."""

    def predict(self, text: str) -> LanguagePrediction: ...


class LinguaLanguageIdentifier:
    """Adapter around ``lingua-language-detector`` using a curated language set."""

    def __init__(self) -> None:
        from lingua import Language, LanguageDetectorBuilder

        languages = [getattr(Language, name) for name in SUPPORTED_LANGUAGE_NAMES]
        self._detector = LanguageDetectorBuilder.from_languages(*languages).build()

    def predict(self, text: str) -> LanguagePrediction:
        confidence_values = self._detector.compute_language_confidence_values(text)
        if not confidence_values:
            raise ValueError("Lingua returned no confidence values")

        # Lingua's own per-language probability, unmodified: the value it
        # assigns to the leading candidate out of the curated language set,
        # not a derived margin over the runner-up.
        best_match = confidence_values[0]
        language_code = best_match.language.iso_code_639_1.name.lower()
        return LanguagePrediction(
            language=language_code,
            confidence=float(best_match.value),
        )


class LanguageDetector:
    """Apply short-text and confidence policies around one reusable backend."""

    def __init__(
        self,
        config: LanguageDetectionConfig | None = None,
        *,
        backend: LanguageIdentifier | None = None,
    ) -> None:
        self.config = config or LanguageDetectionConfig()
        self._backend = backend
        if self._backend is None:
            try:
                self._backend = LinguaLanguageIdentifier()
            except Exception:
                logger.exception("Language detector initialization failed")

    def detect(self, text: str | None) -> LanguageDetectionMetadata:
        """Detect one text safely without retaining or logging its content."""

        if not text or not text.strip():
            return self._uncertain()

        alphabetic_count = sum(character.isalpha() for character in text)
        if alphabetic_count < self.config.min_alphabetic_characters:
            return self._uncertain()

        if self._backend is None:
            return self._failed()

        try:
            prediction = self._backend.predict(text)
            if not self._is_valid_prediction(prediction):
                logger.error("Language detector returned an invalid prediction")
                return self._failed()
        except Exception:
            logger.exception("Language detection execution failed")
            return self._failed()

        confidence = round(
            float(prediction.confidence), _CONFIDENCE_DECIMAL_PLACES
        )

        if confidence < self.config.confidence_threshold:
            return self._uncertain(confidence=confidence)

        return LanguageDetectionMetadata(
            language=prediction.language,
            confidence=confidence,
            status=LanguageDetectionStatus.DETECTED,
        )

    @staticmethod
    def _is_valid_prediction(prediction: object) -> bool:
        if not isinstance(prediction, LanguagePrediction):
            return False
        confidence = prediction.confidence
        return (
            bool(_ISO_639_1_PATTERN.fullmatch(prediction.language))
            and not isinstance(confidence, bool)
            and isinstance(confidence, (int, float))
            and math.isfinite(confidence)
            and 0.0 <= confidence <= 1.0
        )

    @staticmethod
    def _uncertain(confidence: float | None = None) -> LanguageDetectionMetadata:
        return LanguageDetectionMetadata(
            language=None,
            confidence=confidence,
            status=LanguageDetectionStatus.UNCERTAIN,
        )

    @staticmethod
    def _failed() -> LanguageDetectionMetadata:
        return LanguageDetectionMetadata(
            language=None,
            confidence=None,
            status=LanguageDetectionStatus.FAILED,
        )


def build_language_text(subject: str | None, message: str) -> str:
    """Combine already-processed customer text without further cleaning."""

    if subject:
        return f"{subject}\n{message}"
    return message


def annotate_ticket_language(ticket: Ticket, detector: LanguageDetector) -> Ticket:
    """Return a copy of a processed ticket with generated language metadata."""

    detection_text = build_language_text(ticket.subject, ticket.message)
    detection = detector.detect(detection_text)
    return ticket.model_copy(update={"language_detection": detection})
