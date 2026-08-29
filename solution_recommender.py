"""
Hybrid AI Context-Aware Recommendation Engine for HTU SRC System.

Generates two distinct, non-duplicative recommendations per feedback:
1. Student Recommendation - Simple, practical next steps
2. Admin Recommendation - Stand-alone action plan with investigation steps,
   root causes, responsible departments, priority, and success criteria.

Falls back gracefully when the system cannot determine a clear recommendation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ======================================================================
# DATA MODELS
# ======================================================================

@dataclass
class StudentRecommendation:
    """Simple, actionable recommendation for students."""
    summary: str           # One-line summary of what to do
    action_steps: List[str]  # Practical steps the student can take
    who_to_contact: str    # Department or person to reach out to
    timeline: str          # Expected resolution timeframe
    additional_tips: str   # Helpful advice or context


@dataclass
class AdminRecommendation:
    """Comprehensive action plan for SRC administrators."""
    issue_summary: str     # Clear description of the issue
    investigation_steps: List[str]  # Steps to investigate root cause
    root_cause_analysis: str  # Likely underlying causes
    immediate_actions: List[str]  # What to do right now
    long_term_actions: List[str]  # Systemic improvements
    responsible_departments: List[str]  # Who owns each action
    priority: str          # Critical / High / Medium / Low
    estimated_timeline: str  # How long to resolve
    success_criteria: List[str]  # How to measure resolution
    resources_needed: str  # Budget, personnel, tools required


@dataclass
class Recommendation:
    """Combined recommendation output."""
    category: str
    matched_keywords: List[str]
    confidence: float = 0.0
    secondary_categories: List[dict] = field(default_factory=list)
    source_template_id: Optional[int] = None
    
    # Legacy fields (kept for backward compatibility)
    short_term_solution: str = ""
    long_term_solution: str = ""
    responsible_department: str = ""
    estimated_time: str = ""
    
    # New dual-recommendation system
    student_rec: Optional[StudentRecommendation] = None
    admin_rec: Optional[AdminRecommendation] = None


# ======================================================================
# CONTEXT DETECTION
# ======================================================================

# Feedback type classification
FEEDBACK_TYPES = {
    "complaint": ["bad", "terrible", "worst", "horrible", "awful", "poor", "unacceptable", "disappointed", "frustrated", "angry", "annoyed"],
    "suggestion": ["should", "could", "would be nice", "please consider", "recommend", "suggest", "it would be", "hope that", "wish"],
    "inquiry": ["how", "what", "when", "where", "why", "can i", "is it possible", "do you", "could you tell"],
    "appreciation": ["thank", "thanks", "great", "excellent", "amazing", "wonderful", "good job", "well done", "appreciate"],
    "report": ["reporting", "report", "incident", "occurred", "happened", "witnessed", "observed", "noticed"],
    "policy_statement": ["policy", "policies", "rule", "rules", "regulation", "regulations", "has policy", "have policy", "university has"],
    "factual_statement": ["there is", "there are", "exists", "in place", "currently", "the university", "the school"],
    "safety_concern": ["kidnapping", "kidnapped", "abduct", "shooting", "assault", "rape", "murder", "violence", "weapon", "theft", "stolen"],
    "urgent_issue": ["emergency", "urgent", "immediately", "asap", "critical", "danger", "unsafe", "life threatening"],
}


def detect_feedback_type(text: str) -> str:
    """Detect the type of feedback to tailor recommendations."""
    text_lower = text.lower()
    scores = {}
    
    for fb_type, keywords in FEEDBACK_TYPES.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[fb_type] = score
    
    if not scores:
        return "general"
    
    return max(scores, key=scores.get)


def extract_key_entities(text: str) -> Dict[str, List[str]]:
    """Extract key entities from feedback text."""
    text_lower = text.lower()
    
    entities = {
        "locations": [],
        "departments": [],
        "issues": [],
        "timeframes": [],
    }
    
    location_keywords = ["hall", "hostel", "lab", "library", "canteen", "classroom", "office", 
                        "gate", "parking", "field", "court", "building", "block", "floor",
                        "room", "washroom", "toilet", "kitchen", "restaurant"]
    dept_keywords = ["ict", "academic", "security", "maintenance", "catering", "finance", 
                    "registry", "administration", "health", "counseling", "sports", "transport"]
    timeframe_keywords = ["yesterday", "today", "last week", "last month", "this morning", 
                         "this afternoon", "tonight", "every day", "always", "sometimes", "never"]
    
    for loc in location_keywords:
        if loc in text_lower:
            entities["locations"].append(loc)
    
    for dept in dept_keywords:
        if dept in text_lower:
            entities["departments"].append(dept)
    
    for tf in timeframe_keywords:
        if tf in text_lower:
            entities["timeframes"].append(tf)
    
    return entities


# ======================================================================
# SYNONYM & NORMALIZATION
# ======================================================================

SYNONYM_MAP: Dict[str, List[str]] = {
    "wifi": ["wi-fi", "wireless", "internet", "network", "hotspot"],
    "internet": ["wifi", "wi-fi", "network", "connection", "web", "online", "broadband", "data"],
    "network": ["internet", "wifi", "connection", "connectivity", "lan", "ethernet"],
    "slow": ["lag", "lagging", "buffering", "sluggish", "unresponsive", "stuck"],
    "disconnect": ["disconnected", "disconnecting", "drop", "cut off", "offline", "lost connection"],
    "computer": ["pc", "laptop", "desktop", "system", "machine", "workstation"],
    "lab": ["laboratory", "computer lab", "ict lab"],
    "portal": ["student portal", "website", "online portal", "platform"],
    "server": ["servers", "backend", "system", "database"],
    "room": ["bedroom", "dorm", "dormitory", "hall", "hostel room"],
    "toilet": ["washroom", "restroom", "bathroom", "lavatory", "wc", "latrine"],
    "shower": ["bath", "bathing", "washroom"],
    "water": ["tap", "pipe", "plumbing", "supply", "running water"],
    "leak": ["leaking", "leakage", "drip", "dripping", "pipe burst", "burst"],
    "flood": ["flooded", "flooding", "water logged", "overflow"],
    "lecturer": ["teacher", "professor", "instructor", "tutor", "lecture", "facilitator"],
    "exam": ["examination", "test", "assessment", "quiz", "final"],
    "course": ["subject", "module", "program", "unit", "curriculum"],
    "assignment": ["homework", "coursework", "project", "task", "paper", "report"],
    "grade": ["score", "mark", "result", "gpa", "cgpa", "grading"],
    "timetable": ["schedule", "academic calendar", "class schedule"],
    "food": ["meal", "catering", "canteen food", "dining", "cuisine"],
    "canteen": ["cafeteria", "dining hall", "mess", "food court", "restaurant"],
    "library": ["study area", "reading room", "resource center", "e-library"],
    "classroom": ["lecture hall", "teaching room", "class", "lecture room"],
    "security": ["guard", "security guard", "watchman", "surveillance", "safety"],
    "bus": ["shuttle", "transport", "vehicle", "trotro", "campus bus"],
    "stress": ["stressed", "anxiety", "anxious", "overwhelmed", "pressure", "burnout", "depression"],
    "fee": ["fees", "school fees", "payment", "tuition", "charges", "financial", "invoice"],
    "broken": ["damaged", "faulty", "not working", "malfunctioning", "out of order", "defective"],
    "delay": ["delayed", "postponed", "late", "overdue", "behind schedule"],
    "clean": ["cleanliness", "hygiene", "sanitary", "sanitation", "tidy", "neat"],
    "dirty": ["filthy", "messy", "unclean", "unsanitary", "unkempt"],
    "electricity": ["power", "light", "electrical", "generator", "current"],
    "malpractice": ["cheating", "exam fraud", "impersonation", "leakage"],
}

WORD_NORMALIZATION: Dict[str, str] = {
    "disconnecting": "disconnect", "disconnected": "disconnect",
    "lagging": "lag", "sluggish": "slow", "unresponsive": "slow",
    "flooded": "flood", "flooding": "flood",
    "leaking": "leak", "leakage": "leak", "dripping": "leak",
    "damaged": "broken", "malfunctioning": "broken", "defective": "broken",
    "stressed": "stress", "anxious": "anxiety", "overwhelmed": "stress",
    "delayed": "delay", "overdue": "delay",
    "unsanitary": "dirty", "filthy": "dirty", "messy": "dirty",
}


# ======================================================================
# SOLUTION TEMPLATES
# ======================================================================

SOLUTION_TEMPLATES: Dict[str, List[dict]] = {
    "ICT/Wi-Fi": [
        {"keywords": ["wifi", "wi-fi", "internet", "network", "slow", "disconnect", "offline", "no connection", "portal", "server", "lag", "buffering", "bandwidth", "signal", "speed", "connectivity", "hotspot"], "short_term_solution": "Please disconnect and reconnect to the Wi-Fi, and restart your device. If the issue persists, report the exact time and location so ICT can trace the network logs.", "long_term_solution": "ICT should review access point performance, check authentication/server logs, and apply firmware/config tuning. Add additional coverage in affected areas if signal weakness is confirmed.", "responsible_department": "ICT / Network Support", "estimated_time": "1-3 days"},
        {"keywords": ["lab", "computer", "pc", "laptop", "desktop", "system", "login", "password", "authentication", "access"], "short_term_solution": "Try a different browser/device and clear cache. Ensure your credentials are active and report the error text (if any).", "long_term_solution": "Investigate authentication/portal service health and update user provisioning if needed. Add monitoring/alerts for failed login spikes.", "responsible_department": "ICT / Systems", "estimated_time": "2-5 days"},
        {"keywords": ["email", "outlook", "student email", "mail", "inbox", "send", "receive"], "short_term_solution": "Clear your browser cache and try accessing via a different device or browser. Check if your password has expired.", "long_term_solution": "ICT should verify email server health, check mailbox quotas, and ensure SMTP/IMAP services are running correctly.", "responsible_department": "ICT / Email Support", "estimated_time": "1-2 days"},
    ],
    "Accommodation": [
        {"keywords": ["room", "bed", "mattress", "cupboard", "furniture", "furnished"], "short_term_solution": "Report the specific item and room number to Hall Management for immediate assessment.", "long_term_solution": "Hall Management should conduct a room-by-room inventory and schedule repairs/replacements during semester breaks.", "responsible_department": "Hall Management", "estimated_time": "2-5 days"},
        {"keywords": ["toilet", "washroom", "restroom", "bathroom", "shower", "lavatory", "wc"], "short_term_solution": "Report the affected bathroom location to Hall Management. Temporary cleaning will be arranged.", "long_term_solution": "Maintenance should inspect plumbing fixtures and replace failing parts. Introduce preventive maintenance schedules and quick-response checklists for recurring failures.", "responsible_department": "Hall Management / Maintenance", "estimated_time": "2-4 days"},
        {"keywords": ["leak", "leaking", "flood", "flooded", "water", "pipe", "plumbing", "drip", "dripping", "burst", "overflow", "drainage", "sewage"], "short_term_solution": "Report the affected location to Hall Management immediately. If there is active leakage/flooding, request an immediate cleanup and temporary water control.", "long_term_solution": "Maintenance should inspect plumbing fixtures and replace failing parts. Introduce preventive maintenance schedules and quick-response checklists for recurring failures.", "responsible_department": "Hall Management / Maintenance", "estimated_time": "2-4 days"},
        {"keywords": ["electricity", "power", "light", "socket", "switch", "fan", "generator", "electrical", "current"], "short_term_solution": "Report the specific electrical issue to Hall Management. Avoid using damaged sockets/switches until inspected.", "long_term_solution": "Electrical team should inspect wiring, replace faulty components, and ensure proper load balancing.", "responsible_department": "Hall Management / Electrical", "estimated_time": "1-3 days"},
    ],
    "Academics": [
        {"keywords": ["lecturer", "teacher", "professor", "instructor", "tutor", "facilitator", "lecture"], "short_term_solution": "Report the specific concern to the Department Head. Include dates and details of the incident.", "long_term_solution": "Department should review teaching evaluations, provide faculty development, and establish clear lecturer-student communication channels.", "responsible_department": "Academic Affairs / Department", "estimated_time": "3-10 days"},
        {"keywords": ["exam", "examination", "test", "assessment", "quiz", "result", "grade", "score", "mark", "gpa", "cgpa", "grading", "malpractice", "cheating"], "short_term_solution": "Contact your Department or the Academic Office to verify your results. You may request a re-check within the specified period.", "long_term_solution": "The Academic Board should review assessment policies, ensure timely release of results, and implement a transparent grade appeal process.", "responsible_department": "Academic Affairs / Examinations", "estimated_time": "1-2 weeks"},
        {"keywords": ["course", "subject", "module", "curriculum", "syllabus", "program", "timetable", "schedule", "semester", "academic calendar"], "short_term_solution": "Check with your Department or Academic Advisor for clarification on course registration and scheduling.", "long_term_solution": "Academic Affairs should review curriculum delivery, course scheduling, and ensure timely updates to the academic calendar.", "responsible_department": "Academic Affairs / Department", "estimated_time": "1-2 weeks"},
        {"keywords": ["assignment", "homework", "coursework", "project", "paper", "report", "submission", "deadline"], "short_term_solution": "Contact your lecturer or Department for clarification on submission requirements and deadlines.", "long_term_solution": "Department should establish clear assignment submission guidelines and ensure fair and consistent grading practices.", "responsible_department": "Academic Affairs / Department", "estimated_time": "3-7 days"},
    ],
    "Catering": [
        {"keywords": ["food", "meal", "canteen", "cafeteria", "dining", "kitchen", "restaurant", "cook", "menu", "breakfast", "lunch", "dinner"], "short_term_solution": "Report the specific issue to the Catering Manager. If food quality is a concern, request a meeting with the food service provider.", "long_term_solution": "Catering services should review menu variety, food quality standards, and gather regular student feedback for continuous improvement.", "responsible_department": "Catering Services", "estimated_time": "1-3 days"},
        {"keywords": ["price", "expensive", "cost", "overcharged", "fee", "payment"], "short_term_solution": "Report the pricing concern to the Catering Manager with specific details. Request a price review.", "long_term_solution": "Catering should review pricing structure, compare with other institutions, and consider student input in menu pricing decisions.", "responsible_department": "Catering Services / Finance", "estimated_time": "1-2 weeks"},
        {"keywords": ["hygiene", "clean", "dirty", "filthy", "sanitary", "sanitation", "unsanitary", "messy", "rodent", "cockroach", "infestation", "pest"], "short_term_solution": "Report the hygiene concern immediately to the Catering Manager. The affected area will be cleaned and inspected.", "long_term_solution": "Catering should implement strict hygiene protocols, regular health inspections, and pest control measures.", "responsible_department": "Catering Services / Health & Safety", "estimated_time": "1-3 days"},
    ],
    "Facilities": [
        {"keywords": ["library", "study area", "reading room", "resource center", "e-library", "book", "borrow"], "short_term_solution": "Report the issue to the Library staff. They will assist with any immediate concerns.", "long_term_solution": "Library management should review resource availability, study space allocation, and upgrade facilities as needed.", "responsible_department": "Library Services", "estimated_time": "1-2 weeks"},
        {"keywords": ["classroom", "lecture hall", "lecture room", "class", "teaching room", "projector", "board", "speaker", "microphone", "air conditioner", "ac", "fan", "lighting", "chair", "desk"], "short_term_solution": "Report the specific classroom issue to the Facilities Office. If equipment is faulty, request an alternative room.", "long_term_solution": "Facilities should conduct regular classroom audits, maintain teaching equipment, and fix broken furniture/amenities promptly.", "responsible_department": "Facilities / Maintenance", "estimated_time": "2-5 days"},
        {"keywords": ["elevator", "lift", "broken", "faulty", "stuck", "not working", "out of order"], "short_term_solution": "Report the faulty elevator immediately to Facilities. Post warning signs and secure the area until repairs are done.", "long_term_solution": "Facilities should schedule regular elevator maintenance, inspections, and quick-response repair contracts.", "responsible_department": "Facilities / Maintenance", "estimated_time": "1-3 days"},
        {"keywords": ["sports", "gym", "football", "basketball", "court", "field", "auditorium", "lounge", "parking"], "short_term_solution": "Report the facility issue to the Sports & Recreation Office for assessment.", "long_term_solution": "Facilities should review sports infrastructure, schedule regular maintenance, and upgrade equipment as needed.", "responsible_department": "Sports & Recreation", "estimated_time": "1-2 weeks"},
    ],
    "Safety": [
        {"keywords": ["security", "safe", "safety", "guard", "patrol", "gate", "cctv", "camera", "surveillance", "watchman", "intruder", "suspicious", "vandalism"], "short_term_solution": "Report the security concern to Campus Security immediately. If there is an active threat, contact emergency services.", "long_term_solution": "Security should review patrol schedules, upgrade surveillance systems, and improve lighting in vulnerable areas.", "responsible_department": "Campus Security", "estimated_time": "1-3 days"},
        {"keywords": ["theft", "stolen", "robbery", "burglary", "break-in", "missing"], "short_term_solution": "Report the incident to Campus Security immediately and file a formal report. Secure your personal belongings.", "long_term_solution": "Security should increase patrols, install additional cameras, and conduct student awareness campaigns on personal property safety.", "responsible_department": "Campus Security", "estimated_time": "1-5 days"},
        {"keywords": ["dark", "lighting", "light", "dim", "unlit", "poor lighting", "visibility", "shadow"], "short_term_solution": "Report the dark area to Facilities for immediate lighting repairs. Avoid walking alone in poorly lit areas at night.", "long_term_solution": "Facilities should conduct a campus-wide lighting audit and install additional lighting in identified dark spots.", "responsible_department": "Facilities / Security", "estimated_time": "1-7 days"},
        {"keywords": ["fight", "violence", "assault", "harassment", "threat", "threatened", "attack", "panic", "danger", "unsafe", "hazard", "emergency", "gunshots", "shooting", "weapon", "hostage", "armed", "blood", "injured", "injury", "fire"], "short_term_solution": "EMERGENCY - Contact Campus Security and/or emergency services immediately. Do not intervene directly. Ensure your safety first.", "long_term_solution": "Security should conduct emergency response drills, improve incident reporting systems, and work with local law enforcement for rapid response.", "responsible_department": "Campus Security / Emergency Response", "estimated_time": "Immediate - 1 day"},
    ],
    "Transport": [
        {"keywords": ["bus", "shuttle", "transport", "vehicle", "trotro", "campus bus", "driver", "route", "schedule", "delay"], "short_term_solution": "Report the transport issue to the Transport Office with specific details about the route and time.", "long_term_solution": "Transport services should review route efficiency, driver conduct, and schedule reliability. Consider student feedback for route adjustments.", "responsible_department": "Transport Services", "estimated_time": "3-7 days"},
        {"keywords": ["parking", "car", "vehicle", "park", "space", "parking lot"], "short_term_solution": "Report the parking concern to the Transport Office. Temporary parking arrangements may be available.", "long_term_solution": "Transport should review parking allocation, consider additional parking spaces, and enforce parking regulations fairly.", "responsible_department": "Transport Services / Security", "estimated_time": "1-4 weeks"},
    ],
    "Mental Health": [
        {"keywords": ["stress", "stressed", "anxiety", "anxious", "overwhelmed", "pressure", "burnout", "exhausted", "tired", "worried", "scared", "afraid", "depression", "depressed", "homesick", "lonely", "isolated"], "short_term_solution": "Please reach out to the Counselling Centre for immediate support. You can also speak with a trusted friend, lecturer, or hall warden.", "long_term_solution": "The University should strengthen mental health awareness programs, provide accessible counselling services, and create peer support networks.", "responsible_department": "Counselling Centre / Health Services", "estimated_time": "1-3 days"},
        {"keywords": ["counseling", "counselling", "therapy", "therapist", "mental", "wellness", "wellbeing", "support", "health center", "clinic", "hospital", "doctor", "nurse", "medicine"], "short_term_solution": "Contact the Counselling Centre or Health Services to schedule an appointment. Emergency support is available for urgent cases.", "long_term_solution": "The University should expand mental health services, reduce wait times, and integrate mental wellness into the student experience.", "responsible_department": "Counselling Centre / Health Services", "estimated_time": "1-5 days"},
    ],
    "Financial": [
        {"keywords": ["fee", "fees", "school fees", "payment", "tuition", "charges", "invoice", "receipt", "financial", "money", "pay", "cost", "expensive"], "short_term_solution": "Visit the Finance Office with your student ID and relevant documents for clarification on your fees.", "long_term_solution": "Finance should review fee structures, improve payment options, and ensure transparent communication about fees and charges.", "responsible_department": "Finance Office", "estimated_time": "3-7 days"},
        {"keywords": ["scholarship", "bursary", "financial aid", "sponsorship", "grant", "funding", "support"], "short_term_solution": "Contact the Scholarships Office to inquire about available financial aid options and application procedures.", "long_term_solution": "The University should expand scholarship opportunities, streamline the application process, and provide timely updates to applicants.", "responsible_department": "Scholarships / Finance", "estimated_time": "1-4 weeks"},
    ],
    "Administration": [
        {"keywords": ["admin", "administration", "registry", "registrar", "office", "staff", "secretary", "reception", "front desk", "help desk", "helpdesk", "service", "complaint"], "short_term_solution": "Visit the relevant administrative office with your student ID. If the issue is not resolved, escalate to the Head of Department.", "long_term_solution": "Administration should review service delivery processes, improve response times, and implement a tracking system for student inquiries.", "responsible_department": "Administration / Registry", "estimated_time": "3-10 days"},
        {"keywords": ["transcript", "certificate", "document", "letter", "verification", "confirmation", "admission", "enrollment", "enrolment"], "short_term_solution": "Submit your request at the Academic Affairs or Registry office. Check the processing time and follow up if delayed.", "long_term_solution": "Registry should digitize document requests, reduce processing times, and provide online tracking for applications.", "responsible_department": "Registry / Academic Affairs", "estimated_time": "1-3 weeks"},
    ],
    "Cleanliness": [
        {"keywords": ["clean", "cleanliness", "hygiene", "sanitary", "sanitation", "tidy", "neat", "sweep", "mop", "janitor", "cleaner", "housekeeping", "trash", "rubbish", "garbage", "waste", "bin", "dump", "neglected", "unkempt"], "short_term_solution": "Report the specific location to the Housekeeping or Facilities Office for immediate cleaning.", "long_term_solution": "Management should implement regular cleaning schedules, increase waste bins, and conduct cleanliness awareness campaigns.", "responsible_department": "Housekeeping / Facilities", "estimated_time": "1-2 days"},
        {"keywords": ["rodent", "cockroach", "infestation", "pest", "mold", "mice", "rat", "insect", "bug", "mosquito"], "short_term_solution": "Report the infestation to Hall Management or Facilities immediately for pest control intervention.", "long_term_solution": "Management should schedule regular pest control treatments, seal entry points, and maintain proper waste disposal practices.", "responsible_department": "Facilities / Pest Control", "estimated_time": "1-5 days"},
    ],
    "Other": [],
}


# ======================================================================
# HELPER FUNCTIONS
# ======================================================================

def _normalize(text: str) -> str:
    """Normalize text for keyword matching."""
    text = text or ""
    return re.sub(r"\s+", " ", text.strip().lower())


def _expand_with_synonyms(text: str) -> str:
    """Expand keywords in text by adding synonym variants."""
    norm = _normalize(text)
    expanded = norm
    for word, synonyms in SYNONYM_MAP.items():
        if word in norm:
            for syn in synonyms:
                if syn not in expanded:
                    expanded += " " + syn
    return expanded


def _fuzzy_match_keyword(text: str, keyword: str) -> bool:
    """Check if keyword matches with fuzzy/typo tolerance."""
    if keyword in text:
        return True
    if len(keyword) <= 3:
        return False
    kw_lower = keyword.lower()
    for word in text.split():
        if len(word) < 3:
            continue
        if abs(len(word) - len(kw_lower)) <= 1:
            diffs = sum(1 for a, b in zip(word, kw_lower) if a != b)
            diffs += abs(len(word) - len(kw_lower))
            if diffs <= max(1, len(kw_lower) // 4):
                return True
    return False


def _normalize_words(text: str) -> str:
    """Apply word normalization to handle common variations."""
    norm = _normalize(text)
    for variant, base in WORD_NORMALIZATION.items():
        norm = norm.replace(variant, base)
    return norm


def _detect_negation(text: str, keyword: str) -> bool:
    """Check if a keyword is negated (e.g., 'no wifi', 'not working')."""
    negation_patterns = [
        r"\bno\s+" + re.escape(keyword) + r"\b",
        r"\bnot\s+(?:any\s+)?" + re.escape(keyword) + r"\b",
        r"\b(?:isn't|is\s+not|ain't)\s+(?:any\s+)?" + re.escape(keyword) + r"\b",
        r"\b(?:don't|do\s+not|doesn't|does\s+not)\s+have\s+" + re.escape(keyword) + r"\b",
        r"\b(?:lack|lacking|lacks)\s+of\s+" + re.escape(keyword) + r"\b",
        r"\bwithout\s+" + re.escape(keyword) + r"\b",
        r"\b(?:no\s+more|never)\s+" + re.escape(keyword) + r"\b",
    ]
    text_lower = text.lower()
    for pattern in negation_patterns:
        if re.search(pattern, text_lower):
            return True
    return False


def _get_urgency_level(urgency_score: Optional[int]) -> str:
    """Map urgency score to level."""
    if urgency_score is None:
        return "medium"
    if urgency_score >= 4:
        return "high"
    elif urgency_score >= 3:
        return "medium"
    else:
        return "low"


def extract_keywords(text: str, candidate_keywords: List[str]) -> List[str]:
    """Enhanced keyword extraction with synonym expansion, normalization, and fuzzy matching."""
    if not text or not candidate_keywords:
        return []
    norm = _normalize(text)
    expanded_text = _expand_with_synonyms(norm)
    normalized_text = _normalize_words(norm)
    found: List[str] = []
    for kw in candidate_keywords:
        if not kw:
            continue
        kw_lower = kw.lower().strip()
        if not kw_lower:
            continue
        if _detect_negation(norm, kw_lower):
            continue
        if kw_lower in norm:
            found.append(kw)
        elif kw_lower in expanded_text:
            found.append(kw)
        elif kw_lower in normalized_text:
            found.append(kw)
        elif len(kw_lower) >= 4 and _fuzzy_match_keyword(norm, kw_lower):
            found.append(kw)
    seen = set()
    out = []
    for x in found:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


# ======================================================================
# DUAL RECOMMENDATION GENERATOR
# ======================================================================

def _generate_student_recommendation(
    text: str,
    category: str,
    feedback_type: str,
    template: Optional[dict],
    entities: Dict[str, List[str]],
    urgency_score: int,
) -> StudentRecommendation:
    """Generate a simple, practical recommendation for students."""
    
    # Build action steps based on feedback type
    action_steps = []
    
    if feedback_type == "complaint":
        action_steps.append("Document the issue with specific details (date, time, location)")
        action_steps.append("Report to the relevant department in person or via the SRC portal")
        if entities["locations"]:
            action_steps.append(f"Specify the location: {', '.join(entities['locations'][:2])}")
    elif feedback_type == "suggestion":
        action_steps.append("Submit your suggestion formally through the SRC feedback system")
        action_steps.append("Include specific details about your proposed improvement")
    elif feedback_type == "inquiry":
        action_steps.append("Contact the relevant office during working hours")
        action_steps.append("Have your student ID ready for verification")
    elif feedback_type == "appreciation":
        action_steps.append("Your feedback has been noted and appreciated")
        action_steps.append("Consider sharing your positive experience with peers")
    elif feedback_type == "safety_concern":
        action_steps.append("Report to Campus Security immediately if there is an active threat")
        action_steps.append("Avoid the area until the issue is resolved")
        action_steps.append("Inform your hall warden or a trusted lecturer")
    else:
        action_steps.append("Submit your feedback through the SRC portal")
        action_steps.append("Include as much detail as possible for faster resolution")
    
    # Determine who to contact
    if template:
        who_to_contact = template.get("responsible_department", "SRC Secretariat")
    elif category == "Safety":
        who_to_contact = "Campus Security"
    elif category == "Academics":
        who_to_contact = "Academic Affairs / Department"
    elif category == "ICT/Wi-Fi":
        who_to_contact = "ICT Support"
    else:
        who_to_contact = "SRC Secretariat"
    
    # Determine timeline
    if urgency_score >= 4:
        timeline = "Immediate (within 24 hours)"
    elif urgency_score >= 3:
        timeline = "Priority (1-3 days)"
    elif template:
        timeline = template.get("estimated_time", "3-7 days")
    else:
        timeline = "7-14 days"
    
    # Build summary
    if template:
        summary = template.get("short_term_solution", "Your feedback has been received and will be reviewed.")
    else:
        summary = f"Your feedback regarding {category.lower()} has been received and will be reviewed by the relevant department."
    
    # Additional tips
    tips = "Keep your feedback ID for follow-up. You will receive a notification when the status changes."
    if feedback_type == "safety_concern":
        tips = "For emergencies, contact Campus Security directly. Always prioritize your personal safety."
    
    return StudentRecommendation(
        summary=summary,
        action_steps=action_steps,
        who_to_contact=who_to_contact,
        timeline=timeline,
        additional_tips=tips,
    )


def _generate_admin_recommendation(
    text: str,
    category: str,
    feedback_type: str,
    template: Optional[dict],
    entities: Dict[str, List[str]],
    urgency_score: int,
) -> AdminRecommendation:
    """Generate a comprehensive action plan for administrators."""
    
    # Issue summary
    issue_summary = f"Student feedback received regarding {category.lower()}"
    if entities["locations"]:
        issue_summary += f" at {', '.join(entities['locations'][:2])}"
    if entities["timeframes"]:
        issue_summary += f" (reported: {entities['timeframes'][0]})"
    issue_summary += f". Feedback type: {feedback_type.replace('_', ' ')}."
    
    # Investigation steps
    investigation_steps = [
        "Review the feedback details and verify the reported information",
        f"Contact the student for additional context if needed",
    ]
    
    if entities["locations"]:
        investigation_steps.append(f"Conduct a site visit to {', '.join(entities['locations'][:2])} to assess the situation")
    
    if feedback_type == "complaint":
        investigation_steps.append("Review relevant records, logs, or previous complaints in the same area")
        investigation_steps.append("Interview relevant staff or witnesses")
    elif feedback_type == "safety_concern":
        investigation_steps.append("Conduct a risk assessment of the reported area/situation")
        investigation_steps.append("Review security footage if applicable")
        investigation_steps.append("Coordinate with Campus Security for immediate assessment")
    
    # Root cause analysis
    if feedback_type == "complaint":
        root_cause = "Likely caused by inadequate service delivery, maintenance gaps, or communication breakdown. Further investigation needed to confirm specific cause."
    elif feedback_type == "safety_concern":
        root_cause = "Potential security infrastructure gap, insufficient patrols, or environmental design vulnerability. Requires thorough risk assessment."
    elif feedback_type == "suggestion":
        root_cause = "Student-identified opportunity for improvement. May indicate systemic gap in current processes or facilities."
    else:
        root_cause = "To be determined through investigation. May involve process gaps, resource constraints, or communication issues."
    
    # Immediate actions
    immediate_actions = [
        "Acknowledge receipt of feedback and assign tracking ID",
        "Categorize and route to the appropriate department",
    ]
    
    if urgency_score >= 4:
        immediate_actions.append("ESCALATE: Immediate attention required - notify department head")
        immediate_actions.append("Implement temporary mitigation measures if safety is at risk")
    
    if feedback_type == "safety_concern":
        immediate_actions.append("Alert Campus Security for immediate assessment")
    
    # Long-term actions
    long_term_actions = []
    if template:
        long_term_actions.append(template.get("long_term_solution", "Review and improve relevant processes and infrastructure"))
    else:
        long_term_actions.append("Conduct a comprehensive review of the reported issue")
        long_term_actions.append("Develop preventive measures to avoid recurrence")
    
    long_term_actions.append("Document lessons learned and update standard operating procedures")
    
    # Responsible departments
    if template:
        responsible = [template.get("responsible_department", "SRC Secretariat")]
    elif category == "Safety":
        responsible = ["Campus Security", "Facilities"]
    elif category == "Academics":
        responsible = ["Academic Affairs", "Department"]
    elif category == "ICT/Wi-Fi":
        responsible = ["ICT Department"]
    else:
        responsible = ["SRC Secretariat"]
    
    # Priority
    if urgency_score >= 5:
        priority = "Critical"
    elif urgency_score >= 4:
        priority = "High"
    elif urgency_score >= 3:
        priority = "Medium"
    else:
        priority = "Low"
    
    # Timeline
    if urgency_score >= 4:
        timeline = "24-48 hours for immediate action; 1-2 weeks for full resolution"
    elif urgency_score >= 3:
        timeline = "3-7 days for initial response; 2-4 weeks for full resolution"
    else:
        timeline = "1-2 weeks for initial response; 1-2 months for full resolution"
    
    # Success criteria
    success_criteria = [
        "Student confirms issue has been resolved or adequately addressed",
        "No similar complaints received within 30 days of resolution",
    ]
    
    if feedback_type == "safety_concern":
        success_criteria.append("Security assessment confirms risk has been mitigated")
        success_criteria.append("Preventive measures have been implemented")
    
    # Resources needed
    if urgency_score >= 4:
        resources = "Immediate staff time, possible emergency budget allocation, inter-departmental coordination"
    else:
        resources = "Staff time for investigation, possible minor budget for repairs/improvements"
    
    return AdminRecommendation(
        issue_summary=issue_summary,
        investigation_steps=investigation_steps,
        root_cause_analysis=root_cause,
        immediate_actions=immediate_actions,
        long_term_actions=long_term_actions,
        responsible_departments=responsible,
        priority=priority,
        estimated_timeline=timeline,
        success_criteria=success_criteria,
        resources_needed=resources,
    )


# ======================================================================
# MAIN RECOMMENDATION FUNCTION
# ======================================================================

def recommend_solutions(
    text: str,
    category: str,
    urgency_score: Optional[int] = None,
    sentiment: Optional[str] = None,
    sentiment_score: Optional[float] = None,
    emotion: Optional[dict] = None,
    db_templates: Optional[List[dict]] = None,
) -> Recommendation:
    """Generate hybrid AI context-aware dual recommendations.
    
    Returns:
        Recommendation with both student_rec and admin_rec populated.
    """
    cat = category or "Other"
    text_lower = text.lower() if text else ""
    urgency = urgency_score or 3
    
    # Detect feedback type and entities
    feedback_type = detect_feedback_type(text)
    entities = extract_key_entities(text)
    
    # Handle positive feedback
    if sentiment == "Positive":
        student_rec = StudentRecommendation(
            summary="Thank you for your positive feedback! We appreciate your kind words.",
            action_steps=["Your feedback has been noted", "Consider sharing your positive experience with peers"],
            who_to_contact="SRC Secretariat",
            timeline="N/A",
            additional_tips="We will continue to maintain and improve our services.",
        )
        admin_rec = AdminRecommendation(
            issue_summary=f"Positive feedback received regarding {category.lower()}.",
            investigation_steps=["Acknowledge the positive feedback", "Identify what worked well to replicate"],
            root_cause="Positive student experience - identify strengths to maintain.",
            immediate_actions=["Record feedback for staff recognition", "Share positive feedback with relevant department"],
            long_term_actions=["Document best practices", "Use as a model for other areas"],
            responsible_departments=[cat if cat != "Other" else "SRC Secretariat"],
            priority="Low",
            estimated_timeline="N/A",
            success_criteria=["Positive trend maintained", "Best practices documented"],
            resources_needed="Minimal - staff time for documentation",
        )
        return Recommendation(
            category=cat, matched_keywords=[], confidence=1.0,
            short_term_solution=student_rec.summary,
            long_term_solution="",
            responsible_department="SRC Secretariat",
            estimated_time="N/A",
            student_rec=student_rec,
            admin_rec=admin_rec,
        )
    
    # Handle neutral feedback with context awareness
    if sentiment == "Neutral":
        # Check for policy statements
        policy_keywords = ["policy", "policies", "rule", "rules", "regulation", "regulations",
                          "has policy", "have policy", "university has", "university have"]
        is_policy = any(kw in text_lower for kw in policy_keywords)
        
        # Check for safety discussion
        safety_keywords = ["kidnapping", "kidnapped", "abduct", "shooting", "assault",
                          "rape", "murder", "violence", "weapon", "terrorist"]
        is_safety = any(kw in text_lower for kw in safety_keywords)
        
        if is_policy:
            student_rec = StudentRecommendation(
                summary="Thank you for highlighting this policy.",
                action_steps=[
                    "If you have concerns about enforcement, provide specific details",
                    "Contact the SRC if you believe the policy needs review",
                ],
                who_to_contact="SRC Secretariat",
                timeline="N/A",
                additional_tips="Policies are regularly reviewed. Your input can help improve them.",
            )
            admin_rec = AdminRecommendation(
                issue_summary=f"Student referenced university policy regarding {category.lower()}.",
                investigation_steps=[
                    "Review the referenced policy for clarity and accessibility",
                    "Assess whether students are adequately informed about this policy",
                ],
                root_cause="Student may be seeking clarification or suggesting policy review.",
                immediate_actions=["Verify the policy is up-to-date and accessible", "Note any suggestions for improvement"],
                long_term_actions=["Consider policy awareness campaign if needed", "Review policy based on student feedback"],
                responsible_departments=["SRC Secretariat", "Administration"],
                priority="Low",
                estimated_timeline="2-4 weeks for review",
                success_criteria=["Policy clarity improved", "Student awareness increased"],
                resources_needed="Staff time for policy review",
            )
        elif is_safety:
            student_rec = StudentRecommendation(
                summary="Thank you for raising awareness about this important safety topic.",
                action_steps=[
                    "If you have specific safety concerns, report them to Campus Security",
                    "Participate in safety awareness programs",
                ],
                who_to_contact="Campus Security / SRC",
                timeline="N/A",
                additional_tips="The SRC takes all safety topics seriously. Your awareness helps keep the community safe.",
            )
            admin_rec = AdminRecommendation(
                issue_summary=f"Student raised safety topic: {category.lower()}.",
                investigation_steps=[
                    "Assess whether this indicates a broader safety concern",
                    "Review current safety measures related to this topic",
                ],
                root_cause="Student may be seeking information or expressing concern about safety.",
                immediate_actions=["Acknowledge the feedback", "Ensure relevant safety information is accessible"],
                long_term_actions=["Consider safety awareness initiatives", "Review safety protocols if needed"],
                responsible_departments=["Campus Security", "SRC"],
                priority="Medium",
                estimated_timeline="1-2 weeks for assessment",
                success_criteria=["Student safety concerns addressed", "Safety information accessible"],
                resources_needed="Staff time for assessment and awareness",
            )
        else:
            student_rec = StudentRecommendation(
                summary="Thank you for your feedback. We will review your comments.",
                action_steps=[
                    "If you have specific concerns, please provide more details",
                    "Contact the SRC if you need a follow-up response",
                ],
                who_to_contact="SRC Secretariat",
                timeline="7-14 days",
                additional_tips="Detailed feedback helps us address issues more effectively.",
            )
            admin_rec = AdminRecommendation(
                issue_summary=f"General feedback received regarding {category.lower()}.",
                investigation_steps=["Review the feedback for actionable items", "Determine if follow-up is needed"],
                root_cause="To be determined - feedback may be informational or suggestive.",
                immediate_actions=["Categorize and file the feedback", "Route to relevant department if actionable"],
                long_term_actions=["Monitor for similar feedback trends", "Address any systemic issues identified"],
                responsible_departments=["SRC Secretariat"],
                priority="Low",
                estimated_timeline="1-2 weeks",
                success_criteria=["Feedback acknowledged", "Actionable items addressed"],
                resources_needed="Minimal staff time",
            )
        
        return Recommendation(
            category=cat, matched_keywords=[], confidence=1.0,
            short_term_solution=student_rec.summary,
            long_term_solution="",
            responsible_department="SRC Secretariat",
            estimated_time="N/A",
            student_rec=student_rec,
            admin_rec=admin_rec,
        )
    
    # Handle negative feedback - find best template match
    templates = db_templates if db_templates else SOLUTION_TEMPLATES.get(cat, SOLUTION_TEMPLATES.get("Other", []))
    
    all_keywords: List[str] = []
    for t in templates:
        all_keywords.extend(t.get("keywords", []))
    
    matched: List[str] = extract_keywords(text, all_keywords)
    
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
    
    # Calculate confidence
    max_possible = max(1, len(best_template.get("keywords", []))) if best_template else 1
    confidence = min(1.0, best_count / max_possible)
    
    # Generate dual recommendations
    student_rec = _generate_student_recommendation(
        text, cat, feedback_type, best_template, entities, urgency
    )
    admin_rec = _generate_admin_recommendation(
        text, cat, feedback_type, best_template, entities, urgency
    )
    
    # Build legacy fields for backward compatibility
    if best_template:
        short_term = best_template.get("short_term_solution", student_rec.summary)
        long_term = best_template.get("long_term_solution", "")
        responsible = best_template.get("responsible_department", "SRC Secretariat")
        estimated = best_template.get("estimated_time", student_rec.timeline)
    else:
        short_term = student_rec.summary
        long_term = ""
        responsible = "SRC Secretariat"
        estimated = student_rec.timeline
    
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
    
    return Recommendation(
        category=cat,
        matched_keywords=matched,
        confidence=round(confidence, 3),
        secondary_categories=secondary_categories[:3],
        short_term_solution=short_term,
        long_term_solution=long_term,
        responsible_department=responsible,
        estimated_time=estimated,
        student_rec=student_rec,
        admin_rec=admin_rec,
    )
