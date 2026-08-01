from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


_BLAME_PATTERNS = [
    r"you\s+always",
    r"you\s+never",
    r"it'?s\s+your\s+fault",
    r"your\s+fault",
    r"because\s+of\s+you",
    r"you\s+did(n\x27t|\s+not)?\b",
    r"stop\s+blaming",
]

_PERSONAL_ATTACK_PATTERNS = [
    r"idiot",
    r"stupid",
    r"useless",
    r"clown",
    r"trash",
    r"worthless",
    r"loser",
    r"moron",
    r"hate\s+you",
    r"shut\s+up",
    r"get\s+out",
]

_FRUSTRATION_PATTERNS = [
    r"this\s+is\s+(so\s+)?ridiculous",
    r"i\s+am\s+(so\s+)?done",
    r"\b(?:fed\s+up|annoyed|frustrated|sick\s+of)\b",
    r"seriously",
    r"unacceptable",
    r"\bno\s+(help|helping)\b",
    r"\bdoesn'?t\s+work\b",
]

_CALL_TO_ACTION_PATTERNS = [
    r"everyone\s+(report|spam|flag)",
    r"let'?s\s+(all\s+)?(report|spam|flag)",
    r"go\s+to\s+admin",
    r"dm\s+them",
    r"tag\s+(him|her|them)",
    r"\breport\b\s+them",
    r"we\s+should\s+report",
]

_URGENCY_PATTERNS = [
    r"urgent",
    r"asap",
    r"immediately",
    r"right\s+now",
    r"now\s+please",
    r"hurry",
    r"critical",
    r"emergency",
    r"gunshots?",
    r"shooting",
    r"weapon",
    r"violence",
    r"hostage",
    r"armed",
    r"attack",
    r"threat",
    r"panic",
    r"blood",
    r"injury|injured",
    r"danger",
    r"hazard",
    r"unsafe",
]

_NEGATIVE_WORDBANK = [
    "hate",
    "worst",
    "terrible",
    "awful",
    "disgusting",
    "never",
    "nothing",
    "worst",
]

_REPEAT_PATTERNS = [
    # repeated character runs are handled by cleaning, but phrase repetition is useful
    r"(\b\w+\b)(?:\s+\1\b){2,}",
]


def _compile_patterns(patterns: List[str]) -> List[re.Pattern]:
    return [re.compile(p, flags=re.IGNORECASE) for p in patterns]


_COMPILATIONS: Dict[str, List[re.Pattern]] = {
    "blame": _compile_patterns(_BLAME_PATTERNS),
    "personal_attack": _compile_patterns(_PERSONAL_ATTACK_PATTERNS),
    "frustration": _compile_patterns(_FRUSTRATION_PATTERNS),
    "call_to_action": _compile_patterns(_CALL_TO_ACTION_PATTERNS),
    "urgency": _compile_patterns(_URGENCY_PATTERNS),
    "negative": [
        re.compile(rf"\b{re.escape(w)}\b", flags=re.IGNORECASE) for w in _NEGATIVE_WORDBANK
    ],
    "repetition": _compile_patterns(_REPEAT_PATTERNS),
}


def detect_escalation_signals(message: str) -> Dict[str, Any]:
    text = (message or "").lower()

    found: Dict[str, List[str]] = {
        "blame": [],
        "personal_attack": [],
        "frustration": [],
        "call_to_action": [],
        "urgency": [],
        "negative_writing": [],
        "repetition": [],
    }

    def _collect(key: str, label: str, pats: List[re.Pattern]):
        for rx in pats:
            if rx.search(text):
                found[key].append(label)

    _collect("blame", "blaming language", _COMPILATIONS["blame"])
    _collect("personal_attack", "personal attack", _COMPILATIONS["personal_attack"])
    _collect("frustration", "frustration", _COMPILATIONS["frustration"])
    _collect("call_to_action", "inciting/call-to-action", _COMPILATIONS["call_to_action"])
    _collect("urgency", "urgency indicator", _COMPILATIONS["urgency"])
    _collect("repetition", "repetition/escalation pattern", _COMPILATIONS["repetition"])

    for w_rx, w in zip(_COMPILATIONS["negative"], _NEGATIVE_WORDBANK):
        if w_rx.search(text):
            found["negative_writing"].append(w)

    signals: List[str] = []
    for k, vals in found.items():
        if not vals:
            continue
        if k == "negative_writing":
            signals.append("negative wording")
        else:
            signals.append(k)

    # de-dupe categories
    return {
        "found": found,
        "signals": list(dict.fromkeys(signals)),
    }


def compute_message_risk(
    sentiment: str | None,
    urgency_score: int | None,
    signals: List[str],
) -> Tuple[int, str, Dict[str, Any]]:
    risk = 0

    if sentiment == "Negative":
        risk += 25
    elif sentiment == "Neutral":
        risk += 5

    u = urgency_score or 1
    risk += int((u - 1) * 10)  # roughly 0..40

    weights = {
        "blame": 20,
        "personal_attack": 30,
        "frustration": 15,
        "call_to_action": 35,
        "urgency": 15,
        "negative_writing": 10,
        "repetition": 10,
    }
    for s in signals:
        risk += weights.get(s, 0)

    risk = max(0, min(100, risk))

    if risk < 30:
        level = "🟢 Low"
    elif risk < 60:
        level = "🟡 Moderate"
    elif risk < 85:
        level = "🔴 High"
    else:
        level = "🔴⚠️ Critical"

    prediction: Dict[str, Any] = {"what_might_happen_next": []}
    if "personal_attack" in signals or "call_to_action" in signals:
        prediction["what_might_happen_next"].append(
            "Thread may deteriorate quickly into direct conflict"
        )
    if "blame" in signals:
        prediction["what_might_happen_next"].append("More blame responses likely")
    if sentiment == "Negative" and u >= 4:
        prediction["what_might_happen_next"].append(
            "Escalation/complaint will likely request admin intervention"
        )
    if "repetition" in signals:
        prediction["what_might_happen_next"].append(
            "Users may repeat grievances to intensify the argument"
        )
    if not prediction["what_might_happen_next"]:
        prediction["what_might_happen_next"].append(
            "Escalation risk remains present unless moderated"
        )

    details = {
        "base_sentiment_risk": (25 if sentiment == "Negative" else 5 if sentiment == "Neutral" else 0),
        "urgency_score": u,
        "signals": signals,
    }

    return risk, level, prediction | {"details": details}

