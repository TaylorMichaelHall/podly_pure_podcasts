import json
import os
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests
from flask import Flask

from app.config_store import _apply_top_level_env_overrides
from app.extensions import db
from app.models import Identification, ModelCall, Post, TranscriptSegment
from app.routes.config_routes import _sanitize_config_for_client
from podcast_processor.ad_classifier import AdClassifier
from podcast_processor.jev_client import (
    MAX_QUESTIONS_PER_REQUEST,
    JevAPIError,
    JevClient,
    JevSegment,
)
from shared.config import Config
from shared.test_utils import create_standard_test_config


def _response(status_code: int, body: Any) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = body
    resp.text = json.dumps(body)
    return resp


def _session_answering(probabilities: dict[int, float]) -> MagicMock:
    """Session whose POST answers seg_<i> with probabilities[i] (default 0.0)."""

    def post(url: str, **kwargs: Any) -> MagicMock:
        questions = kwargs["json"]["questions"]
        answers = {
            key: {"type": "noul", "noul": probabilities.get(int(key[4:]), 0.0)}
            for key in questions
        }
        return _response(200, {"model": "jev-1.13.0", "answers": answers})

    session = MagicMock(spec=requests.Session)
    session.post.side_effect = post
    return session


def _jev_config() -> Config:
    config = create_standard_test_config()
    config.ad_classifier_backend = "jev"
    config.jev_api_key = "ts-test"
    config.enable_boundary_refinement = True
    return config


@pytest.fixture
def app() -> Generator[Flask, None, None]:
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    with app.app_context():
        db.init_app(app)
        db.create_all()
        yield app


@pytest.mark.parametrize(
    ("api_key", "base_url", "model", "expected_url", "expected_model"),
    [
        ("ts-key", None, None, "https://api.typesafe.ai", "jev-latest"),
        (
            "sk-or-v1-abc",
            None,
            None,
            "https://openrouter.ai/api",
            "~typesafe/jev-latest",
        ),
        (
            "sk-or-v1-abc",
            "https://openrouter.ai/api/",
            "typesafe/jev-1.13",
            "https://openrouter.ai/api",
            "typesafe/jev-1.13",
        ),
        (
            "ts-key",
            "https://gw.example/typesafe",
            None,
            "https://gw.example/typesafe",
            "jev-latest",
        ),
    ],
)
def test_client_resolves_provider(
    api_key: str,
    base_url: str | None,
    model: str | None,
    expected_url: str,
    expected_model: str,
) -> None:
    client = JevClient(api_key=api_key, base_url=base_url, model=model)
    assert client.base_url == expected_url
    assert client.model == expected_model


def test_classify_sends_one_noul_per_segment_and_keeps_likely_ads() -> None:
    session = _session_answering({1: 0.97, 2: 0.6, 3: 0.2})
    client = JevClient(api_key="sk-or-v1-abc", session=session)
    segments = [JevSegment(start=float(i * 10), text=f"line {i}") for i in range(4)]

    result = client.classify_ad_segments(
        segments, podcast_title="Show", podcast_description="About things"
    )

    session.post.assert_called_once()
    url = session.post.call_args.args[0]
    kwargs = session.post.call_args.kwargs
    assert url == "https://openrouter.ai/api/v1/systemone"
    assert kwargs["headers"]["Authorization"] == "Bearer sk-or-v1-abc"
    body = kwargs["json"]
    assert body["model"] == "~typesafe/jev-latest"
    assert body["state"]["podcast"]["title"] == "Show"
    assert [t["text"] for t in body["state"]["transcript"]] == [
        "line 0",
        "line 1",
        "line 2",
        "line 3",
    ]
    assert len(body["questions"]) == 4
    question = body["questions"]["seg_1"]
    assert question["type"] == "noul"
    assert question["instructions"]["segment"]["text"] == "line 1"
    assert set(question["criteria"]) == {"true", "false"}

    assert [(p.segment_offset, p.confidence) for p in result.ad_segments] == [
        (10.0, 0.97),
        (20.0, 0.6),
    ]
    assert result.content_type is None


def test_classify_batches_large_chunks() -> None:
    last = MAX_QUESTIONS_PER_REQUEST + 5
    session = _session_answering({0: 0.9, last: 0.9})
    client = JevClient(api_key="ts-key", session=session)
    segments = [JevSegment(start=float(i), text=f"line {i}") for i in range(last + 1)]

    result = client.classify_ad_segments(
        segments, podcast_title=None, podcast_description=None
    )

    assert session.post.call_count == 2
    assert [p.segment_offset for p in result.ad_segments] == [0.0, float(last)]


@pytest.mark.parametrize(
    ("status", "retryable"),
    [(401, False), (422, False), (429, True), (500, True), (529, True)],
)
def test_http_errors_carry_retryability(status: int, retryable: bool) -> None:
    session = MagicMock(spec=requests.Session)
    session.post.return_value = _response(status, {"detail": "nope"})
    client = JevClient(api_key="ts-key", session=session)

    with pytest.raises(JevAPIError) as exc_info:
        client.ping()

    assert exc_info.value.status_code == status
    assert exc_info.value.retryable is retryable


def test_connection_errors_are_retryable() -> None:
    session = MagicMock(spec=requests.Session)
    session.post.side_effect = requests.ConnectionError("boom")
    client = JevClient(api_key="ts-key", session=session)

    with pytest.raises(JevAPIError) as exc_info:
        client.ping()

    assert exc_info.value.retryable


def test_missing_answers_raise() -> None:
    session = MagicMock(spec=requests.Session)
    session.post.return_value = _response(200, {"answers": {}})
    client = JevClient(api_key="ts-key", session=session)

    with pytest.raises(JevAPIError, match="missing answers"):
        client.ping()


def test_jev_backend_disables_llm_boundary_refinement(app: Flask) -> None:
    classifier = AdClassifier(config=_jev_config())
    assert classifier.jev_client is not None
    assert classifier.boundary_refiner is None


def test_llm_backend_does_not_create_jev_client(app: Flask) -> None:
    classifier = AdClassifier(config=create_standard_test_config())
    assert classifier.jev_client is None


def test_process_chunk_with_jev_creates_identifications(app: Flask) -> None:
    config = _jev_config()
    with app.app_context():
        post = Post(
            id=1, feed_id=1, guid="g1", title="Show", download_url="https://x/1.mp3"
        )
        db.session.add(post)
        segments = [
            TranscriptSegment(
                id=i + 1,
                post_id=1,
                sequence_num=i,
                # Past the first 45s so pre-roll look-back doesn't apply
                start_time=float(100 + i * 10),
                end_time=float(110 + i * 10),
                text=f"line {i}",
            )
            for i in range(5)
        ]
        db.session.add_all(segments)
        db.session.commit()

        classifier = AdClassifier(config=config, db_session=db.session)
        assert classifier.jev_client is not None
        classifier.jev_client.session = _session_answering({2: 0.95, 3: 0.75})

        with patch("litellm.completion") as completion:
            classifier._process_chunk(
                chunk_segments=segments,
                system_prompt="unused",
                post=post,
                user_prompt_str="prompt",
            )
            completion.assert_not_called()

        model_call = ModelCall.query.one()
        assert model_call.model_name == "jev:jev-latest"
        assert model_call.status == "success"

        identified = {
            ident.transcript_segment_id: ident.confidence
            for ident in Identification.query.filter_by(label="ad").all()
        }
        # min_confidence in the standard test config is 0.7
        assert identified == {3: 0.95, 4: 0.75}


def test_jev_auth_error_fails_model_call_permanently(app: Flask) -> None:
    with app.app_context():
        model_call = ModelCall(
            post_id=1,
            model_name="jev:jev-latest",
            prompt="p",
            first_segment_sequence_num=0,
            last_segment_sequence_num=0,
            status="pending",
        )
        db.session.add(model_call)
        db.session.commit()

        classifier = AdClassifier(config=_jev_config(), db_session=db.session)
        assert classifier.jev_client is not None
        session = MagicMock(spec=requests.Session)
        session.post.return_value = _response(401, {"detail": "bad key"})
        classifier.jev_client.session = session

        segment = TranscriptSegment(
            post_id=1, sequence_num=0, start_time=0.0, end_time=1.0, text="hi"
        )
        with pytest.raises(JevAPIError):
            classifier._call_model(
                model_call_obj=model_call,
                system_prompt="unused",
                chunk_segments=[segment],
            )

        assert session.post.call_count == 1
        refreshed = db.session.get(ModelCall, model_call.id)
        assert refreshed is not None
        assert refreshed.status == "failed_permanent"


def test_env_overrides_select_jev() -> None:
    config = create_standard_test_config()
    env = {
        "AD_CLASSIFIER_BACKEND": "JEV",
        "JEV_API_KEY": "sk-or-v1-env",
        "JEV_BASE_URL": "https://openrouter.ai/api",
        "JEV_MODEL": "typesafe/jev-1.13",
    }
    with patch.dict(os.environ, env):
        _apply_top_level_env_overrides(config)

    assert config.ad_classifier_backend == "jev"
    assert config.jev_api_key == "sk-or-v1-env"
    assert config.jev_base_url == "https://openrouter.ai/api"
    assert config.jev_model == "typesafe/jev-1.13"


def test_invalid_backend_env_is_ignored() -> None:
    config = create_standard_test_config()
    with patch.dict(os.environ, {"AD_CLASSIFIER_BACKEND": "gpt"}):
        _apply_top_level_env_overrides(config)
    assert config.ad_classifier_backend == "llm"


def test_jev_api_key_is_masked_for_client() -> None:
    sanitized = _sanitize_config_for_client(
        {"llm": {"jev_api_key": "sk-or-v1-secretvalue"}, "whisper": {}}
    )
    assert "jev_api_key" not in sanitized["llm"]
    assert sanitized["llm"]["jev_api_key_preview"] == "sk-o...alue"
