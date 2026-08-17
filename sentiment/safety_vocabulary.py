"""
safety_vocabulary.py

Authoritative, single-source critical-safety vocabulary and detection helpers.

This module is the ONE place that defines which terms represent a genuine
critical safety incident (kidnapping, assault, weapons, shooting, etc.) versus
ordinary safety concerns (harassment, dangerous conditions, injuries).

Every other component (hybrid_engine, sentiment_analyzer, custom_lexicon,
emotion_engine) imports from here so the keyword lists can never drift out of
sync. It also provides token/template-aware matching so that:

- "5 students have been kidnapped"  -> CRITICAL (active incident)
- "The workshop discussed kidnapping" -> NOT critical (passive discussion)

NOTE: Matching intentionally uses a hybrid of exact-token and small-context
heuristics. The same helper is used by the hybrid engine and the VADER-only
chat/forum path so behaviour stays consistent across the whole app.
"""

import re

# ======================================================================
# CRITICAL SAFETY - active incidents that are always urgent emergencies.
# A sentence containing these in an active/incident context MUST be
# treated as Negative with very high urgency (it can never be Neutral).
# ======================================================================
CRITICAL_SAFETY_TERMS = [
    # Kidnapping / abduction
    "kidnap", "kidnapped", "kidnapping", "kidnapper", "kidnappers",
    "abduct", "abducted", "abduction", "hostage", "hostages",
    # Violent / weapons / shooting
    "shooting", "shoot", "shot", "gunshot", "gunshots", "weapon",
    "firearm", "gun", "shooter", "knife", "knives", "machete",
    "cutlass",
    # Stabbing / assault / robbery
    "stabbing", "stabbed", "stab", "assault", "assaulted", "raped",
    "harassed", "harassment",
    "sexual assault", "robbery", "robbed", "mugged", "mugging",
    # Explosives
    "bomb", "explosion", "explode", "exploded",
    # Threatening / attackers
    "threatened", "attacked", "ambushed", "intruder", "intruders",
    # Active campus fire events are critical; discussion/prevention context
    # is filtered by the context helper before this list is applied.
    "fire outbreak", "building fire", "laboratory fire",
]

# ======================================================================
# SAFETY CONCERNS - security-related but NOT automatically an emergency.
# These are negative/high-urgency but should NOT force the critical
# override on their own (e.g. "someone was injured" is serious but not
# necessarily an active incident; harassment is a concern not an attack).
# ======================================================================
SAFETY_CONCERN_TERMS = [
    "violence", "violent", "threat", "threatening", "threatens",
    "harassment", "harassed", "harassing", "injury", "injured",
    "injure",  # lemmatized form of "injured" (preprocessing lemmatizes)
    "danger", "dangerous", "unsafe", "panic", "emergency", "fire",
    "theft", "stolen", "thief", "thieves", "armed", "blood",
    "menacing", "stalker", "lurking", "suspicious", "abuse",
]

# Terms that indicate the safety keyword is being discussed/learned about
# rather than reporting an actual incident. If any of these appear near a
# safety keyword, we do NOT escalate to critical.
_PREVENTION_CONTEXT = [
    "prevented", "prevent", "prevention", "stopped", "stop", "caught",
    "successfully handled", "successfully prevented", "successfully stopped",
    "decision to stop", "decision to prevent", "measures to prevent",
]

_DISCUSSION_CONTEXT = [
    "discussed", "discussing", "workshop", "learned", "learning",
    "lecture", "lectured", "lesson", "prevention", "prevent",
    "awareness", "seminar", "talked", "talk", "training", "taught",
    "preventive", "guidelines", "explained", "explain", "about",
    "policy", "policies", "regulation", "regulations", "rule", "rules",
    "code of conduct", "prohibits", "prohibited", "banned", "bans",
]

# Token based matching: match whole words (case-insensitive) so that
# "KIDNAPPED" and "kidnapped" both match, but "kidnapping" is not
# accidentally triggered by "kidnap" inside a larger unrelated word.
_CRITICAL_TOKEN_RE = re.compile(
    r"\b(?:%s)\b" % "|".join(re.escape(t) for t in CRITICAL_SAFETY_TERMS),
    re.IGNORECASE,
)
_CONCERN_TOKEN_RE = re.compile(
    r"\b(?:%s)\b" % "|".join(re.escape(t) for t in SAFETY_CONCERN_TERMS),
    re.IGNORECASE,
)
_DISCUSSION_TOKEN_RE = re.compile(
    r"\b(?:%s)\b" % "|".join(re.escape(t) for t in _DISCUSSION_CONTEXT),
    re.IGNORECASE,
)
_PREVENTION_TOKEN_RE = re.compile(
    r"\b(?:%s)\b" % "|".join(re.escape(t) for t in _PREVENTION_CONTEXT),
    re.IGNORECASE,
)

# Multi-word phrases that need special handling (e.g. "sexual assault").
_MULTIWORD_CRITICAL = [
    "sexual assault",
]


def has_critical_safety(text: str) -> bool:
    """
    Return True if the text reports a genuine critical safety incident.

    A critical term is treated as a real incident only when it is NOT in a
    passive/discussion context (e.g. "the workshop discussed kidnapping").
    """
    if not text:
        return False

    t = str(text)

    # Multi-word critical phrases first (exact, case-insensitive substring).
    lower = t.lower()
    for phrase in _MULTIWORD_CRITICAL:
        if phrase in lower:
            # Only escalate if not merely discussed.
            if not _is_discussion_context(t):
                return True

    # Only search for critical tokens if there is no discussion context that
    # would neutralise every keyword in the sentence.
    if _is_discussion_context(t) or _is_prevention_context(t):
        return False

    return bool(_CRITICAL_TOKEN_RE.search(t))


def has_safety_concern(text: str) -> bool:
    """
    Return True if the text contains an ordinary safety concern (not a
    critical incident). Used to bump category/urgency without forcing the
    critical override.

    Like has_critical_safety, this respects discussion/prevention context
    so that e.g. "the seminar covered dangerous road-crossing habits" is
    not flagged as a real concern.
    """
    if not text:
        return False
    if _is_discussion_context(str(text)) or _is_prevention_context(str(text)):
        return False
    return bool(_CONCERN_TOKEN_RE.search(str(text)))


def is_prevention_context(text: str) -> bool:
    return _is_prevention_context(text)


def is_discussion_context(text: str) -> bool:
    """
    Public wrapper: True if the text is discussing/learning about a safety
    topic rather than reporting a real experience (e.g. "the workshop
    discussed kidnapping"). Other modules (sentiment_analyzer, hybrid_engine)
    use this to avoid duplicating the same heuristic with different keyword
    lists that can drift out of sync.
    """
    return _is_discussion_context(text)


def _is_prevention_context(text: str) -> bool:
    """Return True when safety language describes prevention or intervention."""
    return bool(_PREVENTION_TOKEN_RE.search(str(text)))


def _is_discussion_context(text: str) -> bool:
    """
    Heuristic: if a discussion/prevention word appears in the same sentence
    as the safety keyword, the safety term is likely being referenced rather
    than experienced. This prevents false positives like
    "the workshop discussed kidnapping."
    """
    return bool(_DISCUSSION_TOKEN_RE.search(str(text)))


def get_critical_terms() -> list:
    """Return the full list of critical terms (for reporting/logging)."""
    return list(CRITICAL_SAFETY_TERMS)


def get_safety_concern_terms() -> list:
    """Return the full list of safety-concern terms."""
    return list(SAFETY_CONCERN_TERMS)
