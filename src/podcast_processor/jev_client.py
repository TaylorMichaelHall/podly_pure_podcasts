"""Ad segment classification with TypeSafe Jev (a System One model).

Jev answers typed questions about a `state` with calibrated probabilities
instead of generating text. We send a transcript chunk as the state and ask one
yes/no ("noul") question per segment: is this segment advertising? The answers
are converted into the same `AdSegmentPredictionList` the LLM path produces, so
everything downstream (identifications, merging, audio cutting) is unchanged.

The same wire protocol is served by TypeSafe directly and by OpenRouter.
See https://docs.typesafe.ai/api.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import requests

from podcast_processor.model_output import AdSegmentPrediction, AdSegmentPredictionList
from shared import defaults as DEFAULTS

# Keep each request well inside Jev's 64k-token budget for state + questions.
MAX_QUESTIONS_PER_REQUEST = 50
MAX_DESCRIPTION_CHARS = 2000
# Predictions below this probability are not recorded in ModelCall.response.
# The configured output.min_confidence still decides what gets cut.
MIN_RECORDED_PROBABILITY = 0.5

AD_QUESTION = (
    "Is `segment` part of an advertisement: a sponsor read, a paid promotion for "
    "an external product or service, or a promo for a different podcast? Use "
    "`transcript` in the state for surrounding context."
)
AD_CRITERIA = {
    "true": (
        "The segment is sponsor or advertising content, including host-read ads, "
        "promo codes, sponsor URLs, promos for other shows, and the short lead-ins "
        "or transitions into and out of an ad break."
    ),
    "false": (
        "The segment is the episode's own content: conversation, interview, news, "
        "or storytelling, the show's own intro and outro, or the hosts promoting "
        "their own work, even when companies or products are discussed."
    ),
}

RETRYABLE_STATUS_CODES = {408, 409, 429, 529}


class JevAPIError(Exception):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code

    @property
    def retryable(self) -> bool:
        if self.status_code is None:
            return True  # connection errors and timeouts
        return self.status_code in RETRYABLE_STATUS_CODES or self.status_code >= 500


@dataclass(frozen=True)
class JevSegment:
    start: float
    text: str


def is_openrouter_key(api_key: str | None) -> bool:
    return bool(api_key) and str(api_key).startswith("sk-or-")


def resolve_base_url(api_key: str | None, base_url: str | None) -> str:
    if base_url:
        return base_url.rstrip("/")
    if is_openrouter_key(api_key):
        return DEFAULTS.JEV_OPENROUTER_BASE_URL
    return DEFAULTS.JEV_DEFAULT_BASE_URL


def resolve_model(base_url: str, model: str | None) -> str:
    if model:
        return model
    if "openrouter.ai" in base_url:
        return DEFAULTS.JEV_OPENROUTER_DEFAULT_MODEL
    return DEFAULTS.JEV_DEFAULT_MODEL


class JevClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = DEFAULTS.JEV_DEFAULT_TIMEOUT_SEC,
        session: requests.Session | None = None,
    ):
        self.api_key = api_key
        self.base_url = resolve_base_url(api_key, base_url)
        self.model = resolve_model(self.base_url, model)
        self.timeout = timeout
        self.session = session or requests.Session()

    def system_one(
        self, state: Any, questions: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """POST /v1/systemone and return the `answers` map."""
        try:
            response = self.session.post(
                f"{self.base_url}/v1/systemone",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={"state": state, "model": self.model, "questions": questions},
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            raise JevAPIError(f"Jev request failed: {e}") from e

        if response.status_code != 200:
            raise JevAPIError(
                f"Jev API error {response.status_code}: {response.text[:500]}",
                status_code=response.status_code,
            )

        try:
            answers = response.json()["answers"]
        except (ValueError, KeyError, TypeError) as e:
            raise JevAPIError(f"Unexpected Jev response: {response.text[:500]}") from e
        missing = set(questions) - set(answers)
        if missing:
            raise JevAPIError(f"Jev response missing answers for {sorted(missing)}")
        return answers

    def ping(self) -> None:
        self.system_one(
            "Thanks to our sponsor for supporting the show.",
            {"probe": {"type": "noul", "instructions": "Does this mention a sponsor?"}},
        )

    def classify_ad_segments(
        self,
        segments: Sequence[JevSegment],
        *,
        podcast_title: str | None,
        podcast_description: str | None,
    ) -> AdSegmentPredictionList:
        """Ask Jev, per segment, whether it is advertising content."""
        state = {
            "podcast": {
                "title": podcast_title or "",
                "description": (podcast_description or "")[:MAX_DESCRIPTION_CHARS],
            },
            "transcript": [
                {"start_seconds": round(seg.start, 1), "text": seg.text}
                for seg in segments
            ],
        }

        predictions: list[AdSegmentPrediction] = []
        for batch_start in range(0, len(segments), MAX_QUESTIONS_PER_REQUEST):
            batch = segments[batch_start : batch_start + MAX_QUESTIONS_PER_REQUEST]
            questions = {
                f"seg_{batch_start + i}": {
                    "type": "noul",
                    "instructions": {
                        "segment": {
                            "start_seconds": round(seg.start, 1),
                            "text": seg.text,
                        },
                        "question": AD_QUESTION,
                    },
                    "criteria": AD_CRITERIA,
                }
                for i, seg in enumerate(batch)
            }
            answers = self.system_one(state, questions)
            for i, seg in enumerate(batch):
                probability = float(answers[f"seg_{batch_start + i}"]["noul"])
                if probability >= MIN_RECORDED_PROBABILITY:
                    predictions.append(
                        AdSegmentPrediction(
                            segment_offset=seg.start,
                            confidence=min(max(probability, 0.0), 1.0),
                        )
                    )

        return AdSegmentPredictionList(ad_segments=predictions)
