"""Recommendation Learning System

Tracks the effectiveness of solution recommendations over time and
provides data-driven improvements to recommendation quality.

Features:
- Track keyword effectiveness vs resolution rate
- Adjust estimated_time based on historical data
- Department assignment learning from admin behavior
- Template performance scoring
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple


class RecommendationLearner:
    """Learns from past recommendations to improve future ones.

    Uses data from SolutionFeedback and Feedback models to:
    1. Identify which keywords/categories lead to quick resolutions
    2. Adjust estimated times based on actual resolution times
    3. Learn which departments actually handle which issues
    4. Score template effectiveness
    """

    def __init__(self, db_session=None):
        self.db = db_session

    # ==================== KEYWORD EFFECTIVENESS ====================

    def calculate_keyword_effectiveness(
        self, feedback_items: List[Any]
    ) -> Dict[str, float]:
        """Calculate how effective each keyword is at predicting speedy resolution.

        Returns dict of keyword -> effectiveness score (0.0 to 1.0).
        Higher scores mean keywords associated with faster resolutions.
        """
        if not feedback_items:
            return {}

        keyword_stats: Dict[str, Dict[str, float]] = {}

        for f in feedback_items:
            keywords = (getattr(f, "recommended_keywords", None) or "").split(",")
            keywords = [k.strip().lower() for k in keywords if k.strip()]

            resolved_at = getattr(f, "resolved_at", None)
            created_at = getattr(f, "created_at", None)
            status = getattr(f, "status", "")

            for kw in keywords:
                if kw not in keyword_stats:
                    keyword_stats[kw] = {
                        "total": 0, "resolved": 0,
                        "total_time": 0.0, "resolved_count": 0,
                    }
                keyword_stats[kw]["total"] += 1

                if status == "Resolved":
                    keyword_stats[kw]["resolved"] += 1
                    if resolved_at and created_at:
                        hours = (resolved_at - created_at).total_seconds() / 3600
                        keyword_stats[kw]["total_time"] += hours
                        keyword_stats[kw]["resolved_count"] += 1

        # Calculate effectiveness score
        effectiveness = {}
        for kw, stats in keyword_stats.items():
            if stats["total"] == 0:
                continue
            resolution_rate = stats["resolved"] / stats["total"]
            avg_time = (
                stats["total_time"] / stats["resolved_count"]
                if stats["resolved_count"] > 0
                else float("inf")
            )
            # Score combines resolution rate with time bonus
            time_bonus = max(0, 1.0 - (avg_time / 168.0)) if avg_time != float("inf") else 0  # 168 hours = 1 week
            effectiveness[kw] = round(resolution_rate * 0.7 + time_bonus * 0.3, 3)

        return effectiveness

    # ==================== ESTIMATED TIME ADJUSTMENT ====================

    def adjust_estimated_time(
        self,
        category: str,
        base_estimated_time: str,
        feedback_items: List[Any],
    ) -> str:
        """Adjust estimated time based on historical resolution times for this category.

        Returns adjusted time string like "1-3 days" or data-driven suggestion.
        """
        relevant = [
            f
            for f in feedback_items
            if getattr(f, "category", None) == category
            and getattr(f, "status", None) == "Resolved"
            and getattr(f, "resolved_at", None)
            and getattr(f, "created_at", None)
        ]

        if len(relevant) < 3:
            return base_estimated_time  # Not enough data to adjust

        times_hours = []
        for f in relevant:
            hours = (f.resolved_at - f.created_at).total_seconds() / 3600
            times_hours.append(hours)

        avg_hours = sum(times_hours) / len(times_hours)
        median_hours = sorted(times_hours)[len(times_hours) // 2]

        # Convert hours to readable time string
        if avg_hours < 24:
            return f"{int(avg_hours)}-{int(avg_hours * 1.5)} hours"
        elif avg_hours < 72:
            days = round(avg_hours / 24, 1)
            return f"{int(days)}-{int(days * 1.5)} days"
        elif avg_hours < 168:
            days = round(avg_hours / 24)
            return f"{days}-{days + 2} days"
        else:
            weeks = round(avg_hours / 168, 1)
            return f"{int(weeks)}-{int(weeks + 1)} weeks"

    # ==================== DEPARTMENT LEARNING ====================

    def learn_department_assignments(
        self, feedback_items: List[Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Learn which departments actually resolve which categories.

        Analyzes admin assignment patterns to suggest better department mapping.
        Returns dict of category -> {department, resolution_rate, avg_time}
        """
        if not feedback_items:
            return {}

        dept_map: Dict[str, Dict[str, Any]] = {}

        for f in feedback_items:
            category = getattr(f, "category", None)
            assigned_to = getattr(f, "assigned_to", None) or getattr(f, "responsible_department", None)
            status = getattr(f, "status", "")
            resolved_at = getattr(f, "resolved_at", None)
            created_at = getattr(f, "created_at", None)

            if not category or not assigned_to:
                continue

            if category not in dept_map:
                dept_map[category] = {
                    "departments": {},
                    "total": 0,
                    "resolved": 0,
                    "total_time": 0.0,
                }

            stats = dept_map[category]
            stats["total"] += 1

            if assigned_to not in stats["departments"]:
                stats["departments"][assigned_to] = {
                    "count": 0,
                    "resolved": 0,
                    "total_time": 0.0,
                }
            stats["departments"][assigned_to]["count"] += 1

            if status == "Resolved":
                stats["resolved"] += 1
                stats["departments"][assigned_to]["resolved"] += 1
                if resolved_at and created_at:
                    hours = (resolved_at - created_at).total_seconds() / 3600
                    stats["total_time"] += hours
                    stats["departments"][assigned_to]["total_time"] += hours

        # Determine best department per category
        result = {}
        for cat, stats in dept_map.items():
            best_dept = max(
                stats["departments"].items(),
                key=lambda x: (
                    x[1]["resolved"] / max(1, x[1]["count"]),
                    x[1]["count"],
                ),
                default=(None, {}),
            )
            if best_dept[0]:
                dept_data = best_dept[1]
                avg_time = (
                    dept_data["total_time"] / max(1, dept_data["resolved"])
                    if dept_data["resolved"] > 0
                    else None
                )
                result[cat] = {
                    "best_department": best_dept[0],
                    "assignment_count": dept_data["count"],
                    "resolution_rate": round(
                        dept_data["resolved"] / max(1, dept_data["count"]) * 100, 1
                    ),
                    "avg_resolution_hours": round(avg_time, 1) if avg_time else None,
                    "total_feedback": stats["total"],
                    "total_resolved": stats["resolved"],
                }

        return result

    # ==================== TEMPLATE PERFORMANCE ====================

    def score_templates(
        self,
        solution_templates: List[Any],
        keyword_effectiveness: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Score template effectiveness based on keyword performance.

        Returns templates sorted by effectiveness score (descending).
        """
        scored = []

        for tmpl in solution_templates:
            keywords_str = getattr(tmpl, "keywords", "") or ""
            keywords = [k.strip().lower() for k in keywords_str.split(",") if k.strip()]

            if not keywords:
                continue

            keyword_scores = [
                keyword_effectiveness.get(kw, 0.5) for kw in keywords
            ]
            avg_keyword_score = (
                sum(keyword_scores) / len(keyword_scores) if keyword_scores else 0.5
            )

            usage = getattr(tmpl, "usage_count", 0) or 0
            resolution = getattr(tmpl, "resolution_count", 0) or 0
            resolution_rate = resolution / max(1, usage)

            # Composite score: 50% keyword effectiveness, 50% resolution rate
            effectiveness_score = round(
                avg_keyword_score * 0.5 + resolution_rate * 0.5, 3
            )

            scored.append({
                "template_id": getattr(tmpl, "id", None),
                "category": getattr(tmpl, "category", ""),
                "keywords": keywords,
                "keyword_effectiveness": round(avg_keyword_score, 3),
                "resolution_rate": round(resolution_rate, 3),
                "usage_count": usage,
                "effectiveness_score": effectiveness_score,
            })

        scored.sort(key=lambda x: x["effectiveness_score"], reverse=True)
        return scored

    # ==================== COMPREHENSIVE REPORT ====================

    def generate_report(
        self,
        feedback_items: List[Any],
        solution_templates: List[Any],
    ) -> Dict[str, Any]:
        """Generate a comprehensive learning report.

        Returns unified dict with all learning insights.
        """
        keyword_effectiveness = self.calculate_keyword_effectiveness(feedback_items)
        department_insights = self.learn_department_assignments(feedback_items)
        template_scores = self.score_templates(
            solution_templates, keyword_effectiveness
        )

        return {
            "keyword_effectiveness": dict(
                sorted(
                    keyword_effectiveness.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:20]
            ),
            "department_insights": department_insights,
            "template_scores": template_scores[:10],
            "generated_at": datetime.utcnow().isoformat(),
            "total_feedback_analyzed": len(feedback_items),
            "total_templates_scored": len(template_scores),
        }
