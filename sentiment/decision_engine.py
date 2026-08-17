"""Evidence fusion for the HTU SRC sentiment system.

The engine distinguishes *unavailable* analyzers from genuine neutral evidence.
It also accepts explicit context evidence, so a strong domain phrase such as
"not working" can outweigh several generic engines that simply do not know the
phrase.
"""

from __future__ import annotations

from typing import Dict, List, Optional


ENGINE_WEIGHTS = {
    "vader": 0.24,
    "textblob": 0.12,
    "afinn": 0.12,
    "sentiwordnet": 0.08,
    "custom": 0.18,
    "context": 0.26,
}


def _direction(score: float) -> str:
    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "neutral"


class DecisionEngine:
    """Combine independent evidence without treating missing models as neutral."""

    def combine_named(
        self,
        scores: Dict[str, Optional[float]],
        context_score: float = 0.0,
        context_confidence: float = 0.0,
        context_label: Optional[str] = None,
    ) -> dict:
        available = {k: float(v) for k, v in scores.items() if v is not None}
        context_is_strong = abs(context_score) >= 0.15 and context_confidence >= 0.45

        weighted_items = []
        for name, score in available.items():
            weighted_items.append((name, score, ENGINE_WEIGHTS.get(name, 0.10)))

        if context_is_strong:
            # Context is domain-specific and gets its configured weight.  It is
            # allowed to dominate when it is much clearer than generic models.
            weighted_items.append(("context", context_score, ENGINE_WEIGHTS["context"]))

        total_weight = sum(w for _, _, w in weighted_items)
        if total_weight:
            final_score = sum(score * weight for _, score, weight in weighted_items) / total_weight
        else:
            final_score = 0.0

        # Strong, explicit context phrases are deterministic enough to override
        # a generic ensemble when the ensemble clearly missed the same meaning.
        override = False
        override_reason = None
        if context_is_strong and context_label in {"Positive", "Negative"}:
            ensemble_without_context = self._weighted_score(available)
            if abs(context_score) >= 0.55 and _direction(ensemble_without_context) != _direction(context_score):
                final_score = context_score
                override = True
                override_reason = "strong_domain_context_override"
            elif context_label == "Negative" and context_score <= -0.65 and ensemble_without_context > -0.05:
                final_score = min(context_score, -0.45)
                override = True
                override_reason = "strong_complaint_override"
            elif context_label == "Positive" and context_score >= 0.55 and ensemble_without_context < 0.05:
                final_score = max(context_score, 0.45)
                override = True
                override_reason = "strong_resolution_override"

        final_score = max(-1.0, min(1.0, final_score))
        directions = [_direction(v) for v in available.values()]
        pos = directions.count("positive")
        neg = directions.count("negative")
        neu = directions.count("neutral")
        n = len(directions)
        consensus = max(pos, neg, neu) / n if n else 0.0
        spread = self._spread(list(available.values())) if available else 0.0
        agreement = max(0.0, min(1.0, 0.65 * consensus + 0.35 * (1.0 - spread)))

        dynamic_weights = {}
        if total_weight:
            for name, _, weight in weighted_items:
                dynamic_weights[name] = round(weight / total_weight, 4)

        return {
            "final_score": round(final_score, 4),
            "dynamic_weights": dynamic_weights,
            "agreement_level": round(agreement, 3),
            "strategy": "context_override" if override else "evidence_fusion",
            "per_engine_scores": {k: round(v, 4) for k, v in available.items()},
            "available_engines": list(available.keys()),
            "unavailable_engines": [k for k, v in scores.items() if v is None],
            "context_score": round(context_score, 4),
            "context_confidence": round(context_confidence, 3),
            "context_label": context_label,
            "override_applied": override,
            "override_reason": override_reason,
            "majority_vote_result": {"positive": pos, "negative": neg, "neutral": neu},
        }

    def combine(self, vader=None, textblob=None, afinn=None, sentiwordnet=None, custom=None):
        result = self.combine_named({
            "vader": vader,
            "textblob": textblob,
            "afinn": afinn,
            "sentiwordnet": sentiwordnet,
            "custom": custom,
        })
        return result["final_score"]

    def combine_with_details(self, vader=None, textblob=None, afinn=None, sentiwordnet=None, custom=None, **kwargs):
        return self.combine_named({
            "vader": vader,
            "textblob": textblob,
            "afinn": afinn,
            "sentiwordnet": sentiwordnet,
            "custom": custom,
        }, **kwargs)

    @staticmethod
    def _weighted_score(scores: Dict[str, float]) -> float:
        items = [(v, ENGINE_WEIGHTS.get(k, 0.10)) for k, v in scores.items()]
        if not items:
            return 0.0
        total = sum(w for _, w in items)
        return sum(v * w for v, w in items) / total if total else 0.0

    @staticmethod
    def _spread(values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return min(1.0, variance ** 0.5)

    @staticmethod
    def _get_direction(score: float) -> str:
        return _direction(score)
