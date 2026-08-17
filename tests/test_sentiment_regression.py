"""
tests/test_sentiment_regression.py

Regression suite for the sentiment/urgency/category pipeline. Every case
here corresponds to either a bug that was found and fixed in a real
support session, or a baseline behaviour that must never regress while
fixing something else.

Run with: pytest tests/test_sentiment_regression.py -v

Requires the real project dependencies (vaderSentiment, textblob, afinn,
nltk) to be installed -- these are in requirements.txt and already run
fine in the deployed app, but were not installable in the sandbox this
suite was originally written in (pypi access was blocked there). Every
assertion below was instead verified by stress-testing the decision
logic across a wide range of injected base-engine scores (0.0 up to
+-0.9) to confirm the expected outcome holds regardless of the exact
value VADER/TextBlob/AFINN/SentiWordNet actually return for that text --
see the session history for that verification if you need to re-derive
confidence in a specific case. Run this file for real (`pytest`) after
any change to sentiment/*.py to catch anything that verification missed.
"""

import pytest

from sentiment.hybrid_engine import HybridSentimentEngine
from sentiment_analyzer import process_feedback


@pytest.fixture(scope="module")
def hybrid():
    return HybridSentimentEngine()


def analyze(hybrid, text):
    return hybrid.analyze(text)


# ============================================================
# Critical safety incidents -- must ALWAYS read as Negative,
# urgency 5, category Safety, regardless of phrasing or of what
# the base sentiment engines make of the wording.
# ============================================================
CRITICAL_SAFETY_CASES = [
    "5 students have been kidnapped.",
    "There was a robbery near the hostel.",
    "A student was assaulted.",
    "A student was threatened.",
    "Someone was injured.",
    "There was a shooting on campus.",
    "Someone brought a weapon to campus.",
    "Students are being harassed.",
    "There was a kidnapping on campus.",
    "A student was abducted.",
    "KIDNAPPED!!!",
    "saw someone with a knife near the gate, i dey fear",
]


@pytest.mark.parametrize("text", CRITICAL_SAFETY_CASES)
def test_critical_safety_incidents_are_negative(hybrid, text):
    result = analyze(hybrid, text)
    assert result["sentiment"] == "Negative"


@pytest.mark.parametrize("text", CRITICAL_SAFETY_CASES)
def test_critical_safety_incidents_score_and_category(text):
    result = process_feedback(text)
    assert result["sentiment"] == "Negative"
    assert result["urgency_score"] == 5
    assert result["detected_category"] == "Safety"


# ============================================================
# Discussion context: a safety topic being discussed/taught/
# policy-referenced is NOT an incident report. This was a real
# bug -- these cases used to come back Negative with inflated
# urgency and a Safety category despite not describing anything
# that happened. Regression-fixed this session; must not recur.
# ============================================================
DISCUSSION_CONTEXT_CASES = [
    "The security workshop discussed kidnapping",
    "The university has a policy against violence.",
    "the seminar covered dangerous road-crossing habits",
]


@pytest.mark.parametrize("text", DISCUSSION_CONTEXT_CASES)
def test_discussion_context_is_not_treated_as_incident(text):
    result = process_feedback(text)
    assert result["sentiment"] != "Negative", (
        f"{text!r} should not read as Negative -- it's discussing the "
        f"topic, not reporting an incident. Got: {result}"
    )
    assert result["urgency_score"] < 5, (
        f"{text!r} should not get incident-level urgency. Got: {result}"
    )
    assert result["detected_category"] != "Safety", (
        f"{text!r} should not be auto-categorised as Safety. Got: {result}"
    )


def test_genuinely_negative_discussion_still_registers_negative():
    """
    The discussion-context guard must not become a blanket excuse to
    ignore real negativity -- if the discussion itself is criticised in
    words the custom lexicon recognises, it should still come back
    Negative.
    """
    result = process_feedback("the fire safety workshop was a complete waste of time")
    assert result["sentiment"] == "Negative"


# ============================================================
# Implicit-negative complaint phrases: no single word in these is
# negative-coded on its own ("high", "cost", "fee" are all
# neutral), but the phrase as a whole is a clear complaint. Also
# a real bug this session -- these used to blend down to Neutral
# because the custom lexicon's vote was diluted by four engines
# that recognised nothing and were scored as "confidently neutral".
# ============================================================
IMPLICIT_NEGATIVE_CASES = [
    "High cost of hostels fee",
    "the wifi has been down for days, still not fixed",
    "I emailed the department twice, no response at all",
]


@pytest.mark.parametrize("text", IMPLICIT_NEGATIVE_CASES)
def test_implicit_negative_complaints_are_negative(text):
    result = process_feedback(text)
    assert result["sentiment"] == "Negative"


# ============================================================
# Baseline sentiment sanity checks (from the project's original
# ad-hoc _sentiment_test.py -- folded in here as real assertions).
# ============================================================
BASELINE_CASES = [
    ("The SRC did a great job organizing the event.", "Positive"),
    ("I am very happy with the new facilities.", "Positive"),
    ("Thank you for solving my issue.", "Positive"),
    ("The service was excellent.", "Positive"),
    ("nice hostel, well maintained and clean", "Positive"),
    ("The hostel water supply is terrible.", "Negative"),
    ("The SRC has completely failed us.", "Negative"),
    ("The internet is extremely slow.", "Negative"),
    ("I am disappointed with this service.", "Negative"),
    ("The service is not good.", "Negative"),
    ("I am not happy with the hostel.", "Negative"),
    ("The lecturers are good but the hostel facilities are terrible.", "Negative"),
    ("The meeting is scheduled for Monday.", "Neutral"),
    ("The library opens at 8am.", "Neutral"),
    ("I submitted my form today.", "Neutral"),
    ("the classroom projector is fine, nothing special", "Neutral"),
]


@pytest.mark.parametrize("text,expected", BASELINE_CASES)
def test_baseline_sentiment(hybrid, text, expected):
    result = analyze(hybrid, text)
    assert result["sentiment"] == expected


def test_negation_flips_sentiment(hybrid):
    """'not bad' is a known negation edge case -- specifically tracked
    because negation handling is a known-limited part of the pipeline
    (see 'improve sentiment analysis' recommendations)."""
    result = analyze(hybrid, "The system is not bad.")
    assert result["sentiment"] == "Positive"


# ============================================================
# Context-aware prevention: safety words do not automatically
# mean a negative incident when the sentence describes a
# successful intervention or prevention.
# ============================================================
@pytest.mark.parametrize("text", [
    "Security prevented a robbery near the hostel.",
    "Security caught the boys that attempted to rob a student.",
    "The school administration has made a decision to stop violence.",
    "Security stopped the attack before anyone was hurt.",
])
def test_prevention_context_is_not_misclassified_as_active_incident(text):
    result = process_feedback(text)
    assert result["urgency_score"] < 5
    assert result["sentiment"] in {"Positive", "Neutral"}


def test_student_explanation_is_short_and_plain_language():
    from sentiment_analyzer import build_ai_explanation
    result = process_feedback("I was robbed at my hostel yesterday.")
    explanation = build_ai_explanation(
        "I was robbed at my hostel yesterday.",
        category=result["detected_category"],
    )
    assert explanation["summary"]
    assert explanation["why"]
    assert explanation["recommendation"]
    assert "keyword" not in explanation["why"].lower()
    assert len(explanation["summary"]) <= 260
