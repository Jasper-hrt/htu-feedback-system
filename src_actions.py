from __future__ import annotations

from typing import Any, Dict, List


def risk_level_to_severity(risk_level: str) -> int:
    # 🟢 Low, 🟡 Moderate, 🔴 High, 🔴⚠️ Critical
    if "Critical" in risk_level:
        return 4
    if "High" in risk_level:
        return 3
    if "Moderate" in risk_level:
        return 2
    return 1


def recommend_src_actions_for_escalation(
    *,
    room_name: str | None,
    message_text: str | None,
    risk_score: int,
    risk_level: str,
    escalation_signals: List[str],
    prediction: Dict[str, Any] | None,
) -> Dict[str, Any]:
    severity = risk_level_to_severity(risk_level)

    actions: List[str] = []
    tips: List[str] = []

    # General actions
    actions.append("Review the triggering message and nearby context")
    tips.append("Use a calm, de-escalating tone; avoid blaming individuals")

    if "personal_attack" in escalation_signals:
        actions.append("Moderation: warn the user to stop personal attacks and restore respectful tone")
        actions.append("If repeated, escalate to admin moderation team")

    if "call_to_action" in escalation_signals:
        actions.append("Public response: remind students about community guidelines")
        actions.append("Direct message/notification to the student(s) involved")

    if "blame" in escalation_signals or "frustration" in escalation_signals:
        actions.append("Acknowledge complaint publicly without assigning fault")
        actions.append("Provide a clear next step/status update")

    if "urgency" in escalation_signals:
        actions.append("Route to the responsible department for urgent attention")
        tips.append("Request exact time/location/details to speed up investigation")

    if "repetition" in escalation_signals:
        actions.append("Prevent escalation loops: summarize the issue and offer a resolution path")

    # Risk-level escalations
    if severity == 1:
        actions.append("Low-risk: post a de-escalation reminder in the room")
    elif severity == 2:
        actions.append("Moderate-risk: issue a gentle warning and encourage constructive wording")
    elif severity == 3:
        actions.append("High-risk: notify SRC/admin and consider restricting/locking if it persists")
    else:  # 4
        actions.append("Critical: notify admin moderation immediately and consider temporary room lock")
        actions.append("Initiate rapid response: coordinate with responsible dept and document incident")

    # Recommended message template(s)
    template_lines: List[str] = []
    template_lines.append("Reminder: Please keep the discussion respectful. Personal attacks will be removed.")
    if severity >= 3:
        template_lines.append("SRC/Admin is reviewing the situation now. Please avoid blaming others and share only the facts (time, location, and what happened).")
    else:
        template_lines.append("If you have additional details (time/location), please add them so the right team can respond.")

    return {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "escalation_signals": escalation_signals,
        "room": room_name,
        "prediction": prediction or {},
        "recommended_actions": actions,
        "tips": tips,
        "public_message_template": " ".join(template_lines),
    }

