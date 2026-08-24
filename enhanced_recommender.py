"""
Enhanced Recommendation Engine with ML-style improvements:
1. Historical solution effectiveness tracking
2. Collaborative filtering (similar feedback → proven solutions)
3. Trend detection for emerging issues
4. Dynamic priority scoring
5. Context-aware recommendations
6. Feedback loop learning
"""

from __future__ import annotations
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta


@dataclass
class EnhancedRecommendation:
    category: str
    matched_keywords: List[str]
    short_term_solution: str
    long_term_solution: str
    responsible_department: str
    estimated_time: str
    confidence: float = 0.0
    secondary_categories: List[dict] = field(default_factory=list)
    source_template_id: Optional[int] = None
    priority_score: float = 0.0
    trend_alert: Optional[str] = None
    similar_resolved_count: int = 0
    effectiveness_score: float = 0.0
    escalation_recommended: bool = False
    auto_assign_to: Optional[str] = None


class RecommendationEngine:
    """Enhanced recommendation engine with learning capabilities."""

    def __init__(self):
        self.solution_effectiveness: Dict[str, List[float]] = defaultdict(list)
        self.category_success_rate: Dict[str, float] = defaultdict(lambda: 0.5)
        self.issue_frequency: Counter = Counter()
        self.assignment_history: Dict[str, List[str]] = defaultdict(list)

    def record_solution_feedback(self, category: str, solution_key: str, was_helpful: bool, resolved: bool):
        """Record whether a solution was effective for learning."""
        score = (1.0 if was_helpful else 0.3) * (1.2 if resolved else 0.8)
        self.solution_effectiveness[f"{category}:{solution_key}"].append(score)
        self.category_success_rate[category] = (
            self.category_success_rate.get(category, 0.5) * 0.9 + (1.0 if resolved else 0.0) * 0.1
        )

    def get_effectiveness(self, category: str, solution_key: str) -> float:
        """Get historical effectiveness score for a solution."""
        scores = self.solution_effectiveness.get(f"{category}:{solution_key}", [])
        if not scores:
            return 0.5
        return sum(scores) / len(scores)

    def calculate_priority_score(
        self,
        urgency_score: int,
        sentiment: str,
        emotion: Optional[dict],
        category: str,
        similar_open_count: int = 0,
    ) -> float:
        """Calculate dynamic priority score (0-100)."""
        score = 0.0

        # Urgency contribution (0-40)
        score += (urgency_score or 3) * 8

        # Sentiment contribution (0-20)
        if sentiment == "Negative":
            score += 20
        elif sentiment == "Neutral":
            score += 10

        # Emotion contribution (0-15)
        if emotion:
            dominant = emotion.get("dominant_emotion", "")
            if dominant in ("anger", "frustration", "fear"):
                score += 15
            elif dominant in ("sadness", "anxiety", "disappointment"):
                score += 10

        # Frequency/trend contribution (0-15)
        freq = self.issue_frequency.get(category, 0)
        if freq > 10:
            score += 15
        elif freq > 5:
            score += 10
        elif freq > 2:
            score += 5

        # Similar open issues (0-10)
        if similar_open_count > 5:
            score += 10
        elif similar_open_count > 2:
            score += 5

        return min(100.0, score)

    def detect_trend(self, category: str, window_hours: int = 24) -> Optional[str]:
        """Detect if an issue category is trending (increasing frequency)."""
        freq = self.issue_frequency.get(category, 0)
        if freq >= 10:
            return f"🔴 HIGH: '{category}' issues are surging ({freq} reports). Consider immediate action."
        elif freq >= 5:
            return f"🟡 MEDIUM: '{category}' showing increased reports ({freq} in window)."
        elif freq >= 3:
            return f"🟢 LOW: '{category}' has {freq} reports. Monitor for escalation."
        return None

    def suggest_auto_assign(self, category: str, department: str) -> Optional[str]:
        """Suggest best assignee based on historical resolution success."""
        history = self.assignment_history.get(category, [])
        if not history:
            return None
        counter = Counter(history)
        most_common = counter.most_common(1)
        if most_common:
            return most_common[0][0]
        return None


# Global engine instance
_engine = RecommendationEngine()


def get_engine() -> RecommendationEngine:
    """Get the global recommendation engine instance."""
    return _engine


def update_issue_frequency(category: str):
    """Update issue frequency counter."""
    _engine.issue_frequency[category] += 1


def record_assignment(category: str, assignee: str):
    """Record who was assigned to resolve an issue."""
    _engine.assignment_history[category].append(assignee)


def recommend_enhanced(
    text: str,
    category: str,
    urgency_score: Optional[int] = None,
    sentiment: Optional[str] = None,
    sentiment_score: Optional[float] = None,
    emotion: Optional[dict] = None,
    db_templates: Optional[List[dict]] = None,
    similar_open_count: int = 0,
) -> EnhancedRecommendation:
    """Enhanced recommendation with ML-style improvements."""
    from solution_recommender import (
        SOLUTION_TEMPLATES, extract_keywords, _normalize,
        _get_urgency_level, URGENCY_ADJUSTMENTS,
        _get_sentiment_emotion_adjustment, Recommendation,
    )

    engine = get_engine()
    cat = category or "Other"
    templates = db_templates if db_templates else SOLUTION_TEMPLATES.get(cat, SOLUTION_TEMPLATES.get("Other", []))

    all_keywords: List[str] = []
    for t in templates:
        all_keywords.extend(t.get("keywords", []))

    matched = extract_keywords(text, all_keywords)

    # Find best-matching template
    best_template: Optional[dict] = None
    best_count = 0
    for t in templates:
        keywords = t.get("keywords", [])
        extracted = extract_keywords(text, keywords)
        weighted_count = sum(1.5 if len(kw.split()) > 1 else 1.0 for kw in extracted)
        if weighted_count > best_count:
            best_count = weighted_count
            best_template = t

    if best_template is None and templates:
        best_template = templates[0]
    elif best_template is None:
        return EnhancedRecommendation(
            category=cat, matched_keywords=matched,
            short_term_solution="Your feedback has been received. An SRC representative will review and respond shortly.",
            long_term_solution="The SRC will review this issue and develop an appropriate action plan.",
            responsible_department="SRC Secretariat", estimated_time="3-10 days", confidence=0.0,
        )

    # Confidence with effectiveness boost
    max_possible = max(1, len(best_template.get("keywords", [])))
    base_confidence = min(1.0, best_count / max_possible)

    # Boost confidence based on historical effectiveness
    solution_key = best_template.get("short_term_solution", "")[:50]
    effectiveness = engine.get_effectiveness(cat, solution_key)
    confidence = min(1.0, base_confidence * (0.8 + 0.4 * effectiveness))

    # Sentiment/emotion adjustment
    sentiment_suffix, conf_mult = _get_sentiment_emotion_adjustment(sentiment, emotion)
    confidence = min(1.0, confidence * conf_mult)

    # Urgency adjustment
    urgency_level = _get_urgency_level(urgency_score)
    urgency_adj = URGENCY_ADJUSTMENTS.get(urgency_level, URGENCY_ADJUSTMENTS["medium"])
    confidence = min(1.0, max(0.0, confidence + urgency_adj["confidence_boost"]))

    # Build solutions
    short_term = best_template["short_term_solution"]
    if urgency_adj["short_term_suffix"]:
        short_term += urgency_adj["short_term_suffix"]
    if sentiment_suffix:
        short_term += sentiment_suffix

    estimated = urgency_adj["estimated_time_prefix"] + best_template["estimated_time"]

    # Calculate priority score
    priority = engine.calculate_priority_score(
        urgency_score or 3, sentiment, emotion, cat, similar_open_count
    )

    # Detect trends
    update_issue_frequency(cat)
    trend_alert = engine.detect_trend(cat)

    # Suggest auto-assignment
    department = best_template["responsible_department"]
    auto_assign = engine.suggest_auto_assign(cat, department)

    # Check if escalation is recommended
    escalation = priority >= 75 or (sentiment == "Negative" and (urgency_score or 0) >= 4)

    # Multi-category detection
    secondary_categories = []
    norm_text = _normalize(text)
    for other_cat, other_templates in SOLUTION_TEMPLATES.items():
        if other_cat == cat or other_cat == "Other":
            continue
        for t_other in other_templates:
            other_kw = t_other.get("keywords", [])
            other_matched = extract_keywords(norm_text, other_kw)
            if len(other_matched) >= 2:
                secondary_categories.append({
                    "category": other_cat,
                    "keywords": other_matched[:5],
                    "confidence": min(1.0, len(other_matched) / max(1, len(other_kw))),
                })
                break
        if len(secondary_categories) >= 3:
            break

    return EnhancedRecommendation(
        category=cat, matched_keywords=matched,
        short_term_solution=short_term,
        long_term_solution=best_template["long_term_solution"],
        responsible_department=department,
        estimated_time=estimated,
        confidence=round(confidence, 3),
        secondary_categories=secondary_categories[:3],
        source_template_id=best_template.get("_db_id"),
        priority_score=round(priority, 1),
        trend_alert=trend_alert,
        similar_resolved_count=similar_open_count,
        effectiveness_score=round(effectiveness, 2),
        escalation_recommended=escalation,
        auto_assign_to=auto_assign,
    )


def get_trending_issues() -> List[Dict[str, Any]]:
    """Get current trending issues for admin dashboard."""
    engine = get_engine()
    trends = []
    for category, count in engine.issue_frequency.most_common(10):
        if count >= 3:
            alert = engine.detect_trend(category)
            trends.append({
                "category": category,
                "count": count,
                "alert": alert,
                "success_rate": round(engine.category_success_rate.get(category, 0.5) * 100, 1),
            })
    return trends


def get_department_workload() -> Dict[str, int]:
    """Get current workload distribution across departments."""
    engine = get_engine()
    workload = Counter()
    for category, assignees in engine.assignment_history.items():
        for assignee in assignees:
            workload[assignee] += 1
    return dict(workload)
