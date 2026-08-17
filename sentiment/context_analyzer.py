"""Context-aware, domain-specific sentiment evidence for HTU SRC feedback.

This module intentionally works without external ML/NLTK packages.  It is the
first semantic layer in the sentiment stack and handles the cases that generic
lexicon models routinely miss: negation, resolution, complaint phrases,
contrast words, Ghanaian/HTU colloquialisms, and uncertainty.

It does not try to replace statistical models.  Instead it supplies explicit,
auditable evidence that is fused with whatever sentiment engines are available.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple


@dataclass
class ContextEvidence:
    score: float = 0.0
    confidence: float = 0.0
    phrases: List[str] | None = None
    negations: List[str] | None = None
    resolutions: List[str] | None = None
    contrasts: List[str] | None = None
    slang_normalized: List[str] | None = None
    uncertainty_markers: List[str] | None = None
    reasons: List[str] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# Strong domain phrases.  Phrase evidence deliberately outranks individual
# words because "not working" and "working again" have opposite meanings.
PHRASE_SCORES: Tuple[Tuple[str, float, str], ...] = (
    ("not working", -0.72, "complaint:not_working"),
    ("not functioning", -0.72, "complaint:not_functioning"),
    ("does not work", -0.70, "complaint:does_not_work"),
    ("doesn't work", -0.70, "complaint:does_not_work"),
    ("does not function", -0.70, "complaint:does_not_function"),
    ("doesn't function", -0.70, "complaint:does_not_function"),
    ("is not working", -0.72, "complaint:is_not_working"),
    ("are not working", -0.72, "complaint:are_not_working"),
    ("was not working", -0.70, "complaint:was_not_working"),
    ("were not working", -0.70, "complaint:were_not_working"),
    ("still broken", -0.68, "complaint:still_broken"),
    ("still not working", -0.78, "complaint:still_not_working"),
    ("keeps failing", -0.70, "complaint:keeps_failing"),
    ("keeps crashing", -0.70, "complaint:keeps_crashing"),
    ("stopped working", -0.65, "complaint:stopped_working"),
    ("cannot access", -0.62, "complaint:cannot_access"),
    ("can't access", -0.62, "complaint:cant_access"),
    ("unable to access", -0.62, "complaint:unable_to_access"),
    ("no access", -0.58, "complaint:no_access"),
    ("no water", -0.60, "complaint:no_water"),
    ("no power", -0.60, "complaint:no_power"),
    ("no electricity", -0.60, "complaint:no_electricity"),
    ("too slow", -0.55, "complaint:too_slow"),
    ("very slow", -0.58, "complaint:very_slow"),
    ("not good", -0.45, "negation:not_good"),
    ("not great", -0.42, "negation:not_great"),
    ("not bad", 0.35, "negation:not_bad"),
    ("not terrible", 0.40, "negation:not_terrible"),
    # Resolution / recovery language.
    ("working again", 0.62, "resolution:working_again"),
    ("works again", 0.62, "resolution:works_again"),
    ("fixed now", 0.58, "resolution:fixed_now"),
    ("finally fixed", 0.60, "resolution:finally_fixed"),
    ("has been fixed", 0.58, "resolution:has_been_fixed"),
    ("have been fixed", 0.58, "resolution:have_been_fixed"),
    ("problem solved", 0.60, "resolution:problem_solved"),
    ("issue resolved", 0.60, "resolution:issue_resolved"),
    ("problem has been resolved", 0.62, "resolution:problem_resolved"),
    ("issue has been resolved", 0.62, "resolution:issue_resolved"),
    ("back to normal", 0.55, "resolution:back_to_normal"),
    ("no longer", 0.38, "resolution:no_longer"),
    ("no more", 0.34, "resolution:no_more"),
    ("has stopped", 0.42, "resolution:has_stopped"),
    ("have stopped", 0.42, "resolution:have_stopped"),
    ("has improved", 0.50, "resolution:has_improved"),
    ("have improved", 0.50, "resolution:have_improved"),
    ("thank you", 0.55, "appreciation:thank_you"),
    ("thanks", 0.48, "appreciation:thanks"),
    ("solving my issue", 0.58, "resolution:solving_issue"),
    ("solved my issue", 0.62, "resolution:solved_issue"),
    ("issue is solved", 0.60, "resolution:issue_solved"),
    ("nothing special", 0.0, "context:qualified_neutral"),
    ("nothing to complain about", 0.28, "context:mild_positive"),
)

NEGATION_WORDS = {
    "not", "no", "never", "neither", "without", "isn't", "wasn't", "weren't",
    "aren't", "don't", "doesn't", "didn't", "can't", "cannot", "won't", "wouldn't",
    "couldn't", "shouldn't", "haven't", "hasn't", "hadn't", "hardly", "barely",
}

CONTRAST_WORDS = {"but", "however", "although", "though", "yet", "except", "while"}
UNCERTAINTY_WORDS = {"maybe", "perhaps", "might", "could", "possibly", "seems", "seemingly", "i think", "not sure"}

# Conservative normalization: these are common Ghanaian/HTU forms that can
# otherwise be invisible to standard English sentiment models.
SLANG_NORMALIZATIONS = {
    "no dey work": "not working",
    "no dey function": "not functioning",
    "dey slow": "is slow",
    "too slow paa": "very slow",
    "bad paa": "very bad",
    "good paa": "very good",
    "chale": "friend",
    "charley": "friend",
    "e no work": "it does not work",
    "e no dey work": "it is not working",
    "e dey work": "it is working",
}

# Token-level sentiment words used only for negation/context handling.  The
# full custom lexicon remains the source of truth for the larger vocabulary.
POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "wonderful", "fantastic", "helpful",
    "better", "best", "improved", "fixed", "resolved", "working", "functional",
    "reliable", "clean", "safe", "efficient", "satisfied", "happy", "comfortable",
}
NEGATIVE_WORDS = {
    "bad", "terrible", "horrible", "awful", "poor", "broken", "faulty", "damaged",
    "slow", "late", "delayed", "frustrating", "frustrating", "useless", "dirty",
    "unsafe", "unacceptable", "unhelpful", "failing", "failed", "failure", "crash",
}


def normalize_domain_language(text: str) -> Tuple[str, List[str]]:
    """Normalize known Ghana/HTU colloquialisms without changing the original text."""
    value = str(text or "").strip().lower()
    applied: List[str] = []
    for source, target in sorted(SLANG_NORMALIZATIONS.items(), key=lambda x: len(x[0]), reverse=True):
        pattern = r"(?<!\w)" + re.escape(source) + r"(?!\w)"
        if re.search(pattern, value, flags=re.IGNORECASE):
            value = re.sub(pattern, target, value, flags=re.IGNORECASE)
            applied.append(f"{source} → {target}")
    return value, applied


def _find_phrases(text: str) -> List[Tuple[str, float, str]]:
    # Prefer the longest matching phrase and avoid double-counting nested
    # phrases such as "not working" + "are not working".
    matches = []
    for phrase, score, reason in PHRASE_SCORES:
        for m in re.finditer(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text, re.IGNORECASE):
            matches.append((m.start(), m.end(), phrase, score, reason))
    matches.sort(key=lambda x: (-(x[1] - x[0]), x[0]))
    accepted = []
    occupied = []
    for start, end, phrase, score, reason in matches:
        if any(start < e and end > s for s, e in occupied):
            continue
        accepted.append((phrase, score, reason))
        occupied.append((start, end))
    accepted.sort(key=lambda x: text.lower().find(x[0].lower()))
    return accepted


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def _negation_adjustments(text: str) -> Tuple[float, List[str]]:
    tokens = _tokenize(text)
    adjustments: List[str] = []
    score = 0.0
    for i, token in enumerate(tokens):
        if token not in POSITIVE_WORDS and token not in NEGATIVE_WORDS:
            continue
        window = tokens[max(0, i - 3):i]
        neg = next((w for w in reversed(window) if w in NEGATION_WORDS), None)
        if not neg:
            continue
        if token in POSITIVE_WORDS:
            score -= 0.42
            adjustments.append(f"negation:{neg} {token} → negative")
        else:
            score += 0.42
            adjustments.append(f"negation:{neg} {token} → positive")
    return score, adjustments


def analyze_context(text: str) -> Dict:
    """Return auditable semantic evidence and a context score in [-1, 1]."""
    if not text or not str(text).strip():
        return ContextEvidence().to_dict()

    normalized, slang = normalize_domain_language(text)
    phrase_hits = _find_phrases(normalized)
    neg_score, negations = _negation_adjustments(normalized)

    contrast_hits = [w for w in CONTRAST_WORDS if re.search(r"(?<!\w)" + re.escape(w) + r"(?!\w)", normalized)]
    uncertainty_hits = [w for w in UNCERTAINTY_WORDS if re.search(r"(?<!\w)" + re.escape(w) + r"(?!\w)", normalized)]

    # Resolution language should dominate a raw negative noun such as "flood"
    # when the sentence explicitly says the problem has ended.
    resolution_hits = [p for p, s, r in phrase_hits if r.startswith("resolution:")]
    complaint_hits = [p for p, s, r in phrase_hits if r.startswith("complaint:")]

    scores = [s for _, s, _ in phrase_hits]
    score = sum(scores) + neg_score

    # Contrast increases uncertainty rather than pretending the entire sentence
    # has one simple polarity. The final score remains bounded.
    if contrast_hits:
        score *= 0.90

    # Uncertainty markers reduce confidence, not necessarily sentiment.
    # Keep explicit context evidence strong without allowing overlapping
    # phrase + negation signals to saturate at -1/+1. Safety incidents are
    # handled separately by the safety classifier.
    score = max(-0.85, min(0.85, score))

    reasons = [r for _, _, r in phrase_hits] + negations
    if slang:
        reasons.append("domain:ghana_htu_normalization")
    if contrast_hits:
        reasons.append("context:contrast")
    if uncertainty_hits:
        reasons.append("context:uncertain_language")

    # Clear phrase evidence is high confidence; generic/ambiguous text is not.
    strength = max((abs(s) for _, s, _ in phrase_hits), default=0.0)
    evidence_count = len(phrase_hits) + len(negations)
    confidence = min(0.95, 0.35 + strength * 0.55 + min(evidence_count, 3) * 0.08)
    if uncertainty_hits:
        confidence *= 0.72
    if contrast_hits:
        confidence *= 0.88

    return ContextEvidence(
        score=round(score, 4),
        confidence=round(confidence, 4),
        phrases=[p for p, _, _ in phrase_hits],
        negations=negations,
        resolutions=resolution_hits,
        contrasts=contrast_hits,
        slang_normalized=slang,
        uncertainty_markers=uncertainty_hits,
        reasons=reasons,
    ).to_dict()


def sentiment_from_context(evidence: Dict) -> Optional[str]:
    score = float(evidence.get("score", 0.0) or 0.0)
    confidence = float(evidence.get("confidence", 0.0) or 0.0)
    if confidence < 0.45:
        return None
    if score >= 0.15:
        return "Positive"
    if score <= -0.15:
        return "Negative"
    return None
