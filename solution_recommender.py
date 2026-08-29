from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Recommendation:
    category: str
    matched_keywords: List[str]
    short_term_solution: str
    long_term_solution: str
    responsible_department: str
    estimated_time: str
    confidence: float = 0.0
    secondary_categories: List[dict] = field(default_factory=list)
    source_template_id: Optional[int] = None


# ==================== SYNONYM MAP ====================
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
}

# ==================== WORD NORMALIZATION ====================
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

# ==================== SOLUTION TEMPLATES ====================
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
        {"keywords": ["exam", "examination", "test", "assessment", "quiz", "result", "grade", "score", "mark", "gpa", "cgpa", "grading"], "short_term_solution": "Contact your Department or the Academic Office to verify your results. You may request a re-check within the specified period.", "long_term_solution": "The Academic Board should review assessment policies, ensure timely release of results, and implement a transparent grade appeal process.", "responsible_department": "Academic Affairs / Examinations", "estimated_time": "1-2 weeks"},
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

# ==================== URGENCY ADJUSTMENT MAPS ====================
URGENCY_ADJUSTMENTS = {
    "high": {"estimated_time_prefix": "Priority: ", "short_term_suffix": " This is a high-priority issue and will be addressed urgently.", "confidence_boost": 0.05},
    "medium": {"estimated_time_prefix": "", "short_term_suffix": "", "confidence_boost": 0.0},
    "low": {"estimated_time_prefix": "Scheduled: ", "short_term_suffix": " This will be addressed during the next available maintenance window.", "confidence_boost": -0.05},
}


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


def _get_sentiment_emotion_adjustment(sentiment: Optional[str], emotion: Optional[dict]) -> Tuple[str, float]:
    """Get solution adjustments based on sentiment and emotion. Returns (urgency_suffix, confidence_multiplier)."""
    if not sentiment:
        return "", 1.0
    suffix = ""
    mult = 1.0
    if sentiment == "Negative":
        mult = 1.1
        suffix = " This issue appears to be causing significant distress."
    if sentiment == "Positive":
        mult = 0.9
        suffix = " We appreciate your positive feedback."
    if emotion:
        dominant = emotion.get("dominant_emotion", "")
        if dominant in ("anger", "frustration"):
            suffix = " We understand this is frustrating and will prioritize your concern."
            mult = 1.2
        elif dominant in ("fear", "anxiety"):
            suffix = " We understand this is concerning and will address it as a priority."
            mult = 1.15
        elif dominant in ("sadness", "disappointment"):
            suffix = " We're sorry to hear this and will work to resolve it."
            mult = 1.05
    return suffix, mult


def recommend_solutions(
    text: str,
    category: str,
    urgency_score: Optional[int] = None,
    sentiment: Optional[str] = None,
    sentiment_score: Optional[float] = None,
    emotion: Optional[dict] = None,
    db_templates: Optional[List[dict]] = None,
) -> Recommendation:
    """Enhanced recommendation with multi-category support, sentiment/emotion awareness, and urgency tailoring.

    Args:
        text: The feedback text to analyze
        category: The primary category
        urgency_score: (1-5) urgency score
        sentiment: Sentiment label (Positive/Negative/Neutral)
        sentiment_score: Numeric sentiment score (-1 to 1)
        emotion: Emotion data dict from HybridSentimentEngine
        db_templates: Optional list of templates from DB (SolutionTemplate model)

    Returns:
        Recommendation dataclass with solutions and metadata
    """
    cat = category or "Other"

    # Handle positive feedback (appreciation/thanks)
    if sentiment == "Positive":
        return Recommendation(
            category=cat, matched_keywords=[],
            short_term_solution="Thank you for your positive feedback! We appreciate your kind words and will continue to maintain and improve our services.",
            long_term_solution="",
            responsible_department="SRC Secretariat",
            estimated_time="N/A",
            confidence=1.0,
        )

    # Handle neutral feedback (general comments/suggestions)
    if sentiment == "Neutral":
        return Recommendation(
            category=cat, matched_keywords=[],
            short_term_solution="Thank you for your feedback. We will review your comments and consider them for future improvements.",
            long_term_solution="",
            responsible_department="SRC Secretariat",
            estimated_time="N/A",
            confidence=1.0,
        )

    templates = db_templates if db_templates else SOLUTION_TEMPLATES.get(cat, SOLUTION_TEMPLATES.get("Other", []))

    all_keywords: List[str] = []
    for t in templates:
        all_keywords.extend(t.get("keywords", []))

    matched: List[str] = extract_keywords(text, all_keywords)

    # Find best-matching template with weighted scoring
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
        return Recommendation(
            category=cat, matched_keywords=matched,
            short_term_solution="Your feedback has been received and flagged for manual review. An SRC representative will assess your concern and provide a personalized response within 3-5 business days.",
            long_term_solution="The SRC will investigate this matter, coordinate with the relevant department if needed, and follow up with you directly.",
            responsible_department="SRC Secretariat",
            estimated_time="3-10 days",
            confidence=0.0,
        )

    # Handle low confidence matches (negative feedback with weak template match)
    max_possible = max(1, len(best_template.get("keywords", [])))
    confidence = min(1.0, best_count / max_possible)
    
    if confidence < 0.3 and sentiment == "Negative":
        return Recommendation(
            category=cat, matched_keywords=matched,
            short_term_solution="Your concern has been received and requires specialized attention. The SRC will assign a dedicated representative to investigate and respond to you directly.",
            long_term_solution="Given the unique nature of this issue, the SRC will conduct a thorough review, engage with the appropriate department, and provide you with a detailed action plan.",
            responsible_department="SRC Secretariat",
            estimated_time="5-14 days",
            confidence=round(confidence, 3),
        )

    # Sentiment/emotion adjustment
    sentiment_suffix, conf_mult = _get_sentiment_emotion_adjustment(sentiment, emotion)
    confidence = min(1.0, confidence * conf_mult)

    # Urgency adjustment
    urgency_level = _get_urgency_level(urgency_score)
    urgency_adj = URGENCY_ADJUSTMENTS.get(urgency_level, URGENCY_ADJUSTMENTS["medium"])
    confidence = min(1.0, max(0.0, confidence + urgency_adj["confidence_boost"]))

    # Build final solutions
    short_term = best_template["short_term_solution"]
    if urgency_adj["short_term_suffix"]:
        short_term += urgency_adj["short_term_suffix"]
    if sentiment_suffix:
        short_term += sentiment_suffix

    estimated = urgency_adj["estimated_time_prefix"] + best_template["estimated_time"]

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
        category=cat, matched_keywords=matched,
        short_term_solution=short_term,
        long_term_solution=best_template["long_term_solution"],
        responsible_department=best_template["responsible_department"],
        estimated_time=estimated,
        confidence=round(confidence, 3),
        secondary_categories=secondary_categories[:3],
        source_template_id=best_template.get("_db_id"),
    )
