"""
confidence.py

Calculates how much the sentiment analyzers
agree with one another and provides per-engine
confidence metrics.

Enhanced with:
- Pairwise agreement matrix
- Per-engine reliability tracking
- Uncertainty detection for near-threshold scores
- Weighted consensus scoring
"""

import math
from typing import Dict, List, Tuple


class ConfidenceCalculator:

    # Engine names for the agreement matrix
    ENGINE_NAMES = ["vader", "textblob", "afinn", "sentiwordnet", "custom"]

    def calculate(self, scores: list) -> float:
        """
        Calculate the overall confidence percentage
        from a list of sentiment scores.

        Uses pairwise agreement between all engine pairs
        to determine how much the engines agree.

        Args:
            scores: List of float scores from different engines

        Returns:
            float: Confidence percentage (0.0 to 100.0)
        """
        # Remove any missing values
        valid_scores = [
            score for score in scores
            if score is not None
        ]

        if len(valid_scores) < 2:
            return 0.0

        # =========================================================
        # METHOD 1: Pairwise Agreement Matrix
        # =========================================================
        # For each pair of engines, check if they agree on sentiment
        # direction (positive/negative/neutral) and magnitude

        n = len(valid_scores)
        pairwise_agreements = []

        for i in range(n):
            for j in range(i + 1, n):
                s1 = valid_scores[i]
                s2 = valid_scores[j]

                # Calculate direction agreement
                dir1 = self._get_direction(s1)
                dir2 = self._get_direction(s2)
                direction_agree = 1.0 if dir1 == dir2 else 0.0

                # Calculate magnitude agreement (how close in value)
                max_diff = 2.0  # Max possible diff between -1 and 1
                actual_diff = abs(s1 - s2)
                magnitude_agree = 1.0 - (actual_diff / max_diff)

                # Combined agreement (70% direction, 30% magnitude)
                combined = direction_agree * 0.7 + magnitude_agree * 0.3
                pairwise_agreements.append(combined)

        # Average pairwise agreement
        avg_pairwise = (
            sum(pairwise_agreements) / len(pairwise_agreements)
            if pairwise_agreements else 0.0
        )

        # =========================================================
        # METHOD 2: Score Spread / Variance
        # =========================================================
        # If all engines give similar scores, confidence is high

        mean_score = sum(valid_scores) / n
        variance = sum((s - mean_score) ** 2 for s in valid_scores) / n
        std_dev = math.sqrt(variance) if variance > 0 else 0.0

        # Maximum std_dev for scores in [-1, 1] is ~1.0
        # Lower spread = higher confidence
        spread_confidence = max(0.0, 1.0 - std_dev)

        # =========================================================
        # METHOD 3: Consensus Strength
        # =========================================================
        # Check how many engines agree on the final direction

        directions = [self._get_direction(s) for s in valid_scores]
        pos_count = directions.count("positive")
        neg_count = directions.count("negative")
        neu_count = directions.count("neutral")

        max_consensus = max(pos_count, neg_count, neu_count)
        consensus_ratio = max_consensus / n

        # =========================================================
        # COMBINE METHODS
        # =========================================================
        # Weight: 50% pairwise agreement, 30% spread, 20% consensus
        combined = (
            avg_pairwise * 0.50 +
            spread_confidence * 0.30 +
            consensus_ratio * 0.20
        )

        # Convert to percentage
        confidence = combined * 100.0

        # Penalize for near-threshold scores (uncertainty)
        confidence = self._apply_threshold_penalty(
            confidence, mean_score
        )

        return round(confidence, 2)

    def calculate_per_engine(self, scores: list) -> Dict[str, float]:
        """
        Calculate confidence for each individual engine
        based on how much it agrees with the majority.

        Returns:
            dict: Engine name -> confidence percentage
        """
        valid_scores = [
            s for s in scores if s is not None
        ]
        if not valid_scores:
            return {name: 0.0 for name in self.ENGINE_NAMES[:len(scores)]}

        n = len(valid_scores)
        directions = [self._get_direction(s) for s in valid_scores]

        # Find majority direction
        pos_count = directions.count("positive")
        neg_count = directions.count("negative")
        neu_count = directions.count("neutral")

        majority_dir = max(
            ["positive", "negative", "neutral"],
            key=lambda d: {
                "positive": pos_count,
                "negative": neg_count,
                "neutral": neu_count
            }[d]
        )

        # Calculate mean score for magnitude comparison
        mean_score = sum(valid_scores) / n

        per_engine = {}
        for i, name in enumerate(self.ENGINE_NAMES[:len(scores)]):
            if i >= len(valid_scores):
                per_engine[name] = 0.0
                continue

            score = valid_scores[i]
            direction = directions[i]

            # Direction agreement (0 or 1)
            dir_agree = 1.0 if direction == majority_dir else 0.0

            # Magnitude closeness to mean
            max_diff = 2.0
            mag_closeness = 1.0 - (abs(score - mean_score) / max_diff)

            # Combined (60% direction, 40% magnitude)
            confidence = (dir_agree * 0.6 + mag_closeness * 0.4) * 100.0

            # Bonus for being in the consensus group
            if dir_agree == 1.0:
                consensus_size = max(pos_count, neg_count, neu_count)
                if consensus_size >= 3:
                    confidence *= 1.1  # 10% bonus for being in strong consensus

            per_engine[name] = round(min(confidence, 100.0), 2)

        return per_engine

    def get_agreement_matrix(self, scores: list) -> Dict[str, dict]:
        """
        Build a pairwise agreement matrix showing how often
        each pair of engines agrees.

        Returns:
            dict: {
                "matrix": {engine: {other_engine: agreement%}},
                "summary": {engine: avg_agreement_with_others}
            }
        """
        valid_scores = [s for s in scores if s is not None]
        if len(valid_scores) < 2:
            return {"matrix": {}, "summary": {}}

        n = len(valid_scores)
        names = self.ENGINE_NAMES[:n]

        matrix = {name: {} for name in names}
        summary = {name: [] for name in names}

        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[names[i]][names[j]] = 100.0
                elif j > i:
                    s1 = valid_scores[i]
                    s2 = valid_scores[j]
                    dir1 = self._get_direction(s1)
                    dir2 = self._get_direction(s2)
                    agree = 100.0 if dir1 == dir2 else 0.0

                    matrix[names[i]][names[j]] = agree
                    matrix[names[j]][names[i]] = agree

                    summary[names[i]].append(agree)
                    summary[names[j]].append(agree)

        # Calculate average agreement per engine
        engine_avg = {}
        for name, agrees in summary.items():
            if agrees:
                engine_avg[name] = round(sum(agrees) / len(agrees), 1)
            else:
                engine_avg[name] = 0.0

        return {
            "matrix": matrix,
            "summary": engine_avg
        }

    @staticmethod
    def _get_direction(score: float) -> str:
        """Classify a score as positive, negative, or neutral."""
        if score >= 0.05:
            return "positive"
        elif score <= -0.05:
            return "negative"
        else:
            return "neutral"

    @staticmethod
    def _apply_threshold_penalty(
        confidence: float, mean_score: float
    ) -> float:
        """
        Apply a penalty when the mean score is near
        the ±0.05 threshold (high uncertainty zone).

        Near 0, small changes flip sentiment, so we reduce confidence.
        """
        abs_mean = abs(mean_score)

        # Scores near 0 are uncertain zones
        if abs_mean < 0.05:
            # Very near zero - heavy penalty
            penalty = 0.5
        elif abs_mean < 0.15:
            # Near threshold - moderate penalty
            penalty = 0.8
        elif abs_mean < 0.3:
            # Moderate certainty - slight penalty
            penalty = 0.95
        else:
            # Strong signal - no penalty
            penalty = 1.0

        return confidence * penalty


def calculate_weighted_confidence(
    scores: list,
    per_engine_confidence: dict = None
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate confidence-weighted average score.

    Instead of static weights, each engine's contribution
    is proportional to its per-engine confidence.

    Args:
        scores: List of scores from each engine
        per_engine_confidence: Dict of {engine_name: confidence%}

    Returns:
        Tuple of (weighted_score, {engine: weight_used})
    """
    calculator = ConfidenceCalculator()

    if per_engine_confidence is None:
        per_engine_confidence = calculator.calculate_per_engine(scores)

    valid_scores = [s for s in scores if s is not None]
    if not valid_scores:
        return 0.0, {}

    n = len(valid_scores)
    engine_names = calculator.ENGINE_NAMES[:n]

    # Calculate dynamic weights from per-engine confidence
    weights = []
    total_weight = 0.0
    weight_map = {}

    for i, name in enumerate(engine_names):
        conf = per_engine_confidence.get(name, 50.0)
        # Convert confidence to weight (ensure minimum weight)
        weight = max(conf / 100.0, 0.05)
        weights.append(weight)
        total_weight += weight

    # Normalize weights to sum to 1.0
    if total_weight > 0:
        weights = [w / total_weight for w in weights]
    else:
        weights = [1.0 / n] * n

    # Calculate weighted score
    weighted_score = sum(
        s * w for s, w in zip(valid_scores, weights)
    )

    # Map engine names to their normalized weights
    for i, name in enumerate(engine_names):
        weight_map[name] = round(weights[i], 4)

    # Keep between -1 and 1
    weighted_score = max(-1.0, min(1.0, weighted_score))

    return round(weighted_score, 4), weight_map

