"""
decision_engine.py

Combines the scores from multiple sentiment analyzers
into one final weighted sentiment score.

Enhanced with:
- Confidence-weighted voting (dynamic weights per analysis)
- Majority vote fallback for high-disagreement scenarios
- Threshold tuning based on score distribution
- Per-engine reliability tracking
"""

from typing import Dict, List

from sentiment.confidence import (
    ConfidenceCalculator,
    calculate_weighted_confidence
)


class DecisionEngine:

    def __init__(self):
        # Static base weights (used as fallback if dynamic weights fail)
        self.base_weights = {
            "vader": 0.40,
            "textblob": 0.20,
            "afinn": 0.15,
            "sentiwordnet": 0.15,
            "custom": 0.10
        }
        self.confidence = ConfidenceCalculator()

    def combine(
        self,
        vader: float,
        textblob: float,
        afinn: float,
        sentiwordnet: float,
        custom: float = 0.0
    ) -> float:
        """
        Combine scores using confidence-weighted voting.

        If engines strongly disagree (high variance), falls back
        to majority vote on sentiment direction.
        """
        scores = [vader, textblob, afinn, sentiwordnet, custom]
        valid_scores = [s for s in scores if s is not None]

        if not valid_scores:
            return 0.0

        # Calculate per-engine confidence
        per_engine_confidence = self.confidence.calculate_per_engine(
            valid_scores
        )

        # Detect disagreement level and choose strategy
        disagreement_level = self._calculate_disagreement(valid_scores)

        if disagreement_level > 0.5:
            # HIGH DISAGREEMENT: Use majority vote
            final_score = self._majority_vote(valid_scores)
        else:
            # LOW/MODERATE DISAGREEMENT: Use confidence-weighted voting
            final_score, _ = calculate_weighted_confidence(
                valid_scores, per_engine_confidence
            )

        final_score = max(-1.0, min(1.0, final_score))
        return round(final_score, 4)

    def combine_with_details(
        self,
        vader: float,
        textblob: float,
        afinn: float,
        sentiwordnet: float,
        custom: float = 0.0
    ) -> dict:
        """
        Combine scores and return detailed information about
        the decision process.
        """
        scores_list = [vader, textblob, afinn, sentiwordnet, custom]
        valid_scores = [s for s in scores_list if s is not None]

        if not valid_scores:
            return {
                "final_score": 0.0,
                "dynamic_weights": {},
                "agreement_level": 0.0,
                "strategy": "no_scores",
                "per_engine_confidence": {},
                "majority_vote_result": None
            }

        # Per-engine confidence
        per_engine_confidence = self.confidence.calculate_per_engine(
            valid_scores
        )

        # Disagreement
        disagreement_level = self._calculate_disagreement(valid_scores)
        agreement_level = 1.0 - disagreement_level

        # Choose strategy
        if disagreement_level > 0.5:
            strategy = "majority_vote"
            final_score = self._majority_vote(valid_scores)
            majority_vote_result = self._majority_vote_details(valid_scores)
            # For dynamic weights, use majority vote shares
            engine_names = ConfidenceCalculator.ENGINE_NAMES[:len(valid_scores)]
            dynamic_weights = {
                name: round(count / len(valid_scores), 4)
                for name, count in zip(engine_names, self._get_vote_shares(valid_scores))
            }
        else:
            strategy = "confidence_weighted"
            final_score, dynamic_weights = calculate_weighted_confidence(
                valid_scores, per_engine_confidence
            )
            majority_vote_result = self._majority_vote_details(valid_scores)

        final_score = max(-1.0, min(1.0, final_score))

        return {
            "final_score": round(final_score, 4),
            "dynamic_weights": dynamic_weights,
            "agreement_level": round(agreement_level, 3),
            "strategy": strategy,
            "per_engine_confidence": per_engine_confidence,
            "majority_vote_result": majority_vote_result
        }

    def _calculate_disagreement(self, scores: List[float]) -> float:
        """
        Calculate how much engines disagree (0 = perfect agreement, 1 = total chaos).
        """
        n = len(scores)
        if n < 2:
            return 0.0

        # Score spread
        mean_score = sum(scores) / n
        variance = sum((s - mean_score) ** 2 for s in scores) / n
        std_dev = variance ** 0.5 if variance > 0 else 0.0
        spread_factor = min(1.0, std_dev)

        # Direction disagreement
        directions = [self._get_direction(s) for s in scores]
        pos = directions.count("positive")
        neg = directions.count("negative")
        neu = directions.count("neutral")

        has_pos = pos > 0
        has_neg = neg > 0
        has_neu = neu > 0

        if has_pos and has_neg and not has_neu:
            direction_factor = 0.8
        elif has_pos and has_neg and has_neu:
            direction_factor = 0.9
        elif (has_pos and has_neu) or (has_neg and has_neu):
            direction_factor = 0.4
        else:
            direction_factor = 0.1

        # Combined (50% spread, 50% direction)
        disagreement = spread_factor * 0.5 + direction_factor * 0.5
        return min(1.0, disagreement)

    def _majority_vote(self, scores: List[float]) -> float:
        """
        Use majority voting to determine the final score.
        """
        pos_scores = []
        neg_scores = []
        neu_scores = []

        for s in scores:
            direction = self._get_direction(s)
            if direction == "positive":
                pos_scores.append(s)
            elif direction == "negative":
                neg_scores.append(s)
            else:
                neu_scores.append(s)

        pos_count = len(pos_scores)
        neg_count = len(neg_scores)
        neu_count = len(neu_scores)

# Find winner
        if pos_count > neg_count and pos_count > neu_count:
            return sum(pos_scores) / len(pos_scores) if pos_scores else 0.0
        elif neg_count > pos_count and neg_count > neu_count:
            return sum(neg_scores) / len(neg_scores) if neg_scores else 0.0
        else:
            # Neutral wins or tie
            if pos_count == neg_count and pos_count > neu_count:
                pos_avg = sum(pos_scores) / len(pos_scores) if pos_scores else 0.0
                neg_avg = sum(neg_scores) / len(neg_scores) if neg_scores else 0.0
                return (pos_avg + neg_avg) / 2
            return 0.0

    def _majority_vote_details(self, scores: List[float]) -> Dict[str, int]:
        """Return vote counts for each sentiment."""
        pos = sum(1 for s in scores if self._get_direction(s) == "positive")
        neg = sum(1 for s in scores if self._get_direction(s) == "negative")
        neu = sum(1 for s in scores if self._get_direction(s) == "neutral")
        return {"positive": pos, "negative": neg, "neutral": neu}

    def _get_vote_shares(self, scores: List[float]) -> List[float]:
        """Get the vote share count for each engine's direction."""
        result = []
        for s in scores:
            d = self._get_direction(s)
            if d == "positive":
                result.append(1.0)
            elif d == "negative":
                result.append(1.0)
            else:
                result.append(0.5)
        return result

    @staticmethod
    def _get_direction(score: float) -> str:
        """Classify a score as positive, negative, or neutral."""
        if score >= 0.05:
            return "positive"
        elif score <= -0.05:
            return "negative"
        else:
            return "neutral"
