from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _normalize_text(text: str | None) -> str:
    t = (text or "").strip().lower()
    # lightweight normalization; the app already stores cleaned fields in many places.
    t = " ".join(t.split())
    return t


def _extract_keywords(text: str, *, max_keywords: int = 12) -> List[str]:
    # Rule-based keyword extraction: split by whitespace and keep simple tokens.
    # If you later want better similarity, replace this with n-gram matching.
    tokens = [tok.strip(".,!?;:\"'()[]{}") for tok in (text or "").lower().split()]
    tokens = [tok for tok in tokens if tok and len(tok) >= 4]
    # de-dupe while preserving order
    seen = set()
    out: List[str] = []
    for tok in tokens:
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
        if len(out) >= max_keywords:
            break
    return out


def _jaccard_similarity(a: List[str], b: List[str]) -> float:
    if not a and not b:
        return 0.0
    sa = set(a)
    sb = set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    if union == 0:
        return 0.0
    return inter / union


def _confidence_to_risk_level(confidence: int) -> str:
    # confidence is 0-100
    if confidence >= 90:
        return "🔴⚠️ Critical"
    if confidence >= 75:
        return "🔴 High"
    if confidence >= 55:
        return "🟡 Moderate"
    return "🟢 Low"


def predict_events_from_chat_history(
    *,
    room_messages: List[Any],
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Historical pattern rules computed from chat messages only.

    Expected message fields (ChatMessage):
      - created_at (datetime)
      - sentiment (str: Positive/Negative/Neutral)
      - urgency_score (int)
      - cleaned_message/message (str)

    Returns predictions with explicit `confidence`.
    """
    now = now or datetime.utcnow()
    msgs = room_messages or []

    # Group within a 24h window
    window_24h_start = now - timedelta(hours=24)
    msgs_24h = [m for m in msgs if getattr(m, "created_at", now) >= window_24h_start]

    # Rule 1: Multiple similar complaints in 24 hours -> likely escalate to protest (80%)
    similar_24h_confidence = None
    evidence_similar: List[str] = []

    # Similarity proxy: keyword overlap among message cleaned_message/message
    clusters: List[Tuple[List[str], List[Any]]] = []  # (keywords, items)

    for m in msgs_24h:
        text = getattr(m, "cleaned_message", None) or getattr(m, "message", None) or ""
        kw = _extract_keywords(_normalize_text(text))
        if not kw:
            continue

        assigned = False
        for i, (base_kw, items) in enumerate(clusters):
            sim = _jaccard_similarity(kw, base_kw)
            if sim >= 0.4:
                items.append(m)
                clusters[i] = (base_kw, items)
                assigned = True
                break
        if not assigned:
            clusters.append((kw, [m]))

    # consider any cluster size>=3 as "multiple similar complaints"
    best_cluster = None
    best_size = 0
    for base_kw, items in clusters:
        if len(items) >= 3 and len(items) > best_size:
            best_cluster = (base_kw, items)
            best_size = len(items)

    if best_cluster is not None:
        _, items = best_cluster
        similar_24h_confidence = 80
        evidence_similar = [
            f"Found {len(items)} similar complaint messages in last 24 hours (keyword overlap cluster)."
        ]

    predictions: List[Dict[str, Any]] = []
    if similar_24h_confidence is not None:
        predictions.append(
            {
                "event": "Likely to escalate to protest",
                "confidence": similar_24h_confidence,
                "evidence": evidence_similar,
                "recommended_actions": [
                    "Notify SRC/admin for early moderation intervention",
                    "Post a de-escalation reminder and ask for only verifiable facts",
                    "Prepare follow-up actions if negative messaging continues",
                ],
            }
        )

    # Rule 2: Negative sentiment increasing over 3 days -> dissatisfaction growing (75%)
    # Compute per-day negative rate for last 3 days.
    day_buckets: List[Dict[str, Any]] = []
    for i in range(3):
        day_start = (now - timedelta(days=2 - i)).date()
        day_end = day_start + timedelta(days=1)
        bucket_msgs = [m for m in msgs if getattr(m, "created_at", now) >= datetime.combine(day_start, datetime.min.time()) and getattr(m, "created_at", now) < datetime.combine(day_end, datetime.min.time())]
        total = len(bucket_msgs)
        neg = sum(1 for m in bucket_msgs if getattr(m, "sentiment", None) == "Negative")
        neg_rate = (neg / total) if total > 0 else 0.0
        day_buckets.append({"day": str(day_start), "total": total, "neg": neg, "neg_rate": neg_rate})

    neg_rates = [b["neg_rate"] for b in day_buckets]
    is_increasing = all(neg_rates[i] <= neg_rates[i + 1] for i in range(len(neg_rates) - 1)) and any(neg_rates[i] < neg_rates[i + 1] for i in range(len(neg_rates) - 1))

    if is_increasing:
        evidence_trend = [
            "Negative rate by day: "
            + ", ".join([f"{b['day']}={round(b['neg_rate']*100,1)}%" for b in day_buckets])
        ]
        predictions.append(
            {
                "event": "Dissatisfaction is growing",
                "confidence": 75,
                "evidence": evidence_trend,
                "recommended_actions": [
                    "Schedule proactive check-ins with responsible department",
                    "Increase moderation frequency and summarize recurring issues",
                ],
            }
        )

    # Rule 4: 10+ angry messages in chat -> conflict imminent (85%)
    # Define angry as Negative sentiment + urgency_score >= 4 OR risk_level high indicators.
    angry_window_start = now - timedelta(hours=24)
    angry_msgs = [
        m
        for m in msgs
        if getattr(m, "created_at", now) >= angry_window_start
        and getattr(m, "sentiment", None) == "Negative"
        and _safe_int(getattr(m, "urgency_score", None), 0) >= 4
    ]

    if len(angry_msgs) >= 10:
        predictions.append(
            {
                "event": "Conflict is imminent",
                "confidence": 85,
                "evidence": [f"Detected {len(angry_msgs)} angry messages in the last 24 hours."],
                "recommended_actions": [
                    "Escalate to admin moderation and consider restricting/locking if it persists",
                    "Issue direct community reminder and request specific verifiable details",
                ],
            }
        )

    # No SRC response in 48 hours depends on feedback table; handled by feedback rules.
    return predictions


def predict_events_from_feedback_history(
    *,
    feedback_items: List[Any],
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Historical pattern rules computed from feedback table.

    Expected feedback fields (Feedback):
      - created_at
      - status
      - src_response (text or None)

    Returns predictions with explicit `confidence`.
    """
    now = now or datetime.utcnow()
    items = feedback_items or []

    no_response_candidates: List[Any] = []
    for f in items:
        created_at = getattr(f, "created_at", None)
        if not created_at:
            continue
        age = now - created_at
        if age < timedelta(hours=48):
            continue

        src_response = getattr(f, "src_response", None)
        status = getattr(f, "status", None)

        # Heuristic: if not resolved and src_response empty/None -> "no SRC response"
        responded = bool((src_response or "").strip())
        resolved = status == "Resolved"
        if not responded and not resolved:
            no_response_candidates.append(f)

    if no_response_candidates:
        predictions = [
            {
                "event": "Students may lose trust",
                "confidence": 90,
                "evidence": [
                    f"Found {len(no_response_candidates)} feedback items older than 48 hours without SRC response."
                ],
                "recommended_actions": [
                    "Prioritize SRC follow-up and publish status update templates",
                    "Assign responsible department and set an expected timeline",
                ],
            }
        ]
        return predictions

    return []


def predict_events_combined(
    *,
    room_messages: List[Any],
    feedback_items: List[Any],
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Combine chat + feedback predictions into a unified response.

    Returns:
      {
        'predictions': [...],
        'early_warning_level': '...',
        'max_confidence': int
      }
    """
    now = now or datetime.utcnow()

    chat_preds = predict_events_from_chat_history(room_messages=room_messages, now=now)
    feedback_preds = predict_events_from_feedback_history(feedback_items=feedback_items, now=now)

    predictions = chat_preds + feedback_preds

    max_conf = max((p.get("confidence", 0) for p in predictions), default=0)
    early_level = _confidence_to_risk_level(max_conf)

    # De-dup by event name (keep highest confidence)
    best_by_event: Dict[str, Dict[str, Any]] = {}
    for p in predictions:
        evt = p.get("event") or "Unknown"
        if evt not in best_by_event or p.get("confidence", 0) > best_by_event[evt].get("confidence", 0):
            best_by_event[evt] = p

    deduped = sorted(best_by_event.values(), key=lambda x: x.get("confidence", 0), reverse=True)

    return {
        "predictions": deduped,
        "early_warning_level": early_level,
        "max_confidence": int(max_conf),
    }

