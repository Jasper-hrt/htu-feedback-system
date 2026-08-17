"""Regression tests for the context-aware HTU sentiment architecture."""

import pytest

from sentiment.hybrid_engine import HybridSentimentEngine
from sentiment_analyzer import analyze_chat_message, analyze_sentiment, process_feedback


@pytest.fixture(scope="module")
def engine():
    return HybridSentimentEngine()


@pytest.mark.parametrize("text", [
    "Most computers are not working",
    "Air conditioners are not working",
    "The WiFi is not working",
    "The portal does not work",
    "The projector is still broken",
])
def test_complaint_phrases_are_negative(engine, text):
    r = engine.analyze(text)
    assert r["sentiment"] == "Negative"
    assert r["context"]["phrases"]


@pytest.mark.parametrize("text", [
    "There are no more flood issues",
    "The flood problem has been resolved",
    "The portal is working again",
    "The computers have been fixed",
])
def test_resolution_language_is_not_negative(engine, text):
    r = engine.analyze(text)
    assert r["sentiment"] == "Positive"
    assert r["context"]["resolutions"]


def test_discussion_of_kidnapping_is_not_incident():
    r = process_feedback("The security workshop discussed kidnapping")
    assert r["sentiment"] == "Neutral"
    assert r["urgency_score"] < 5
    assert r["detected_category"] == "Other"


def test_active_kidnapping_is_critical():
    r = process_feedback("A student was kidnapped yesterday")
    assert r["sentiment"] == "Negative"
    assert r["urgency_score"] == 5


def test_active_fire_outbreak_is_critical():
    r = process_feedback("There is a fire outbreak on campus")
    assert r["sentiment"] == "Negative"
    assert r["urgency_score"] == 5


def test_negation_is_handled(engine):
    positive = engine.analyze("The service is not bad")
    negative = engine.analyze("The service is not good")
    assert positive["sentiment"] == "Positive"
    assert negative["sentiment"] == "Negative"


def test_chat_uses_same_engine():
    r = analyze_chat_message("The computers are not working")
    assert r["sentiment"] == "Negative"
    assert r["model_version"] == "HTU-Sentiment-v3-context-fusion"


def test_forum_api_uses_same_engine():
    sentiment, score = analyze_sentiment("The WiFi is not working")
    assert sentiment == "Negative"
    assert score < 0


def test_unavailable_engines_are_not_fake_neutral_votes(engine):
    r = engine.analyze("The computers are not working")
    assert "unavailable_engines" in r
    assert r["context"]["confidence"] >= 0.45


def test_qualified_neutral_stays_neutral(engine):
    r = engine.analyze("The classroom projector is fine, nothing special")
    assert r["sentiment"] == "Neutral"


def test_model_explanation_is_from_hybrid():
    from sentiment_analyzer import get_sentiment_explanation
    r = get_sentiment_explanation("The computers are not working")
    assert r["sentiment_explanation"]
    assert r["model_version"] == "HTU-Sentiment-v3-context-fusion"
