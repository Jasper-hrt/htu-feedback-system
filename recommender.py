from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum


# ==================== CONSTANTS ====================
FALLBACK_CONFIDENCE_THRESHOLD = 0.15


# ==================== ENUMS ====================

class SentimentType(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"
    UNCLEAR = "unclear"


class UrgencyLevel(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ==================== DATA CLASSES ====================

@dataclass
class CategoryMatch:
    """A detected category with confidence and evidence."""
    name: str
    confidence: float  # 0.0 to 1.0
    evidence: List[str]  # phrases/keywords that triggered this category
    is_primary: bool = False


@dataclass
class StudentRecommendation:
    """Simple, relevant guidance for the student."""
    summary: str  # one-line summary of what was understood
    immediate_action: str  # what the student can do right now
    who_to_contact: str  # specific person/office
    expected_timeline: str
    additional_tips: List[str] = field(default_factory=list)


@dataclass
class AdminActionPlan:
    """Investigation, corrective and preventive actions for SRC/admin."""
    investigation_steps: List[str]  # how to verify and understand the issue
    corrective_actions: List[str]  # immediate fixes
    preventive_actions: List[str]  # long-term prevention
    responsible_department: str
    priority_level: str
    estimated_resolution_time: str
    escalation_path: str  # who to escalate to if unresolved
    monitoring_indicators: List[str] = field(default_factory=list)


@dataclass
class RecommendationResult:
    """Complete recommendation result."""
    primary_category: str
    all_categories: List[CategoryMatch]
    sentiment: str
    urgency: str
    student_recommendation: StudentRecommendation
    admin_action_plan: AdminActionPlan
    confidence: float
    fallback_used: bool = False
    fallback_message: Optional[str] = None
    multi_issue: bool = False


# ==================== CATEGORY DEFINITIONS ====================

CATEGORY_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "Academics": {
        "description": "Teaching, learning, curriculum, assessment, and academic quality",
        "strong_indicators": [
            "lecturer", "professor", "teacher", "instructor", "tutor", "facilitator",
            "teaching", "lecture", "class", "tutorial", "seminar", "workshop",
            "exam", "examination", "test", "quiz", "assessment", "evaluation",
            "course", "subject", "module", "curriculum", "syllabus", "program",
            "assignment", "homework", "coursework", "project", "paper", "report",
            "grade", "score", "mark", "result", "gpa", "cgpa", "grading",
            "academic", "education", "learning", "study", "studies",
            "practical", "lab session", "field work", "research",
            "thesis", "dissertation", "defense", "viva",
            "note", "handout", "textbook", "reading material",
            "consultation", "office hours", "academic advisor",
        ],
        "phrase_patterns": [
            r"lecturer\s+(?:is\s+)?(?:absent|late|not\s+coming|did\s+not\s+come)",
            r"(?:no|not)\s+(?:lecturer|teacher|professor)",
            r"class\s+(?:is\s+)?(?:cancelled|canceled|not\s+holding)",
            r"(?:exam|test|quiz)\s+(?:is\s+)?(?:postponed|cancelled|canceled|delayed)",
            r"result\s+(?:is\s+)?(?:not\s+released|delayed|late|missing)",
            r"grade\s+(?:is\s+)?(?:not\s+posted|missing|wrong|unfair)",
            r"assignment\s+(?:is\s+)?(?:not\s+marked|not\s+graded|missing)",
            r"syllabus\s+(?:is\s+)?(?:not\s+covered|incomplete|unfinished)",
            r"(?:not|never)\s+(?:teaching|explaining|giving\s+notes)",
            r"(?:poor|bad|terrible)\s+teaching",
            r"(?:unfair|biased)\s+(?:grading|marking|assessment)",
            r"(?:rushing|fast)\s+(?:syllabus|course|teaching)",
        ],
        "weak_indicators": [
            "student", "education", "university", "school", "campus",
        ],
        "department": "Academic Affairs / Department",
    },
    "ICT": {
        "description": "Information technology, internet, computers, portals, and digital services",
        "strong_indicators": [
            "wifi", "wi-fi", "wireless", "internet", "network", "broadband",
            "computer", "pc", "laptop", "desktop", "system", "machine", "workstation",
            "portal", "student portal", "website", "online portal", "platform",
            "server", "database", "backend", "hosting",
            "email", "outlook", "student email", "mail", "inbox",
            "password", "login", "authentication", "access", "account",
            "printer", "scanner", "projector", "smart board",
            "software", "application", "app", "program", "system",
            "ict", "information technology", "digital", "tech",
            "cyber", "online", "web", "browser",
            "bandwidth", "signal", "speed", "connectivity",
            "hack", "hacked", "breach", "security breach",
        ],
        "phrase_patterns": [
            r"(?:wifi|internet|network)\s+(?:is\s+)?(?:down|not\s+working|slow|unstable|unavailable|broken|terrible|horrible|awful|disconnected|offline)",
            r"(?:no|not)\s+(?:wifi|internet|network|connection)",
            r"(?:portal|website|system|server)\s+(?:is\s+)?(?:down|not\s+working|crashed|unavailable|broken)",
            r"(?:cannot|can't|unable\s+to)\s+(?:login|log\s+in|access|connect|register|submit)",
            r"(?:computer|pc|laptop|system)\s+(?:is\s+)?(?:slow|not\s+working|broken|crashed|frozen|unresponsive)",
            r"(?:printer|projector)\s+(?:is\s+)?(?:not\s+working|broken|faulty|unavailable)",
            r"(?:email|mail)\s+(?:is\s+)?(?:not\s+working|not\s+(?:sending|receiving)|unavailable)",
            r"(?:password|account)\s+(?:is\s+)?(?:not\s+working|expired|locked|blocked|invalid)",
            r"(?:internet|wifi|network|connection)\s+(?:is\s+)?(?:too\s+slow|very\s+slow|extremely\s+slow|disconnecting|dropping|unstable|always\s+disconnecting|keeps\s+disconnecting)",
            r"(?:online\s+registration|online\s+system|student\s+portal|registration\s+system)\s+(?:is\s+)?(?:down|not\s+working|broken|unavailable|slow|efficient|good|excellent|working\s+well)",
            r"(?:network|internet|wifi)\s+(?:in\s+(?:the\s+)?(?:lecture\s+hall|classroom|library|hostel|dorm|lab|lab\s+room))\s+(?:is\s+)?(?:down|not\s+working|slow|disconnecting|unavailable|broken)",
        ],
        "weak_indicators": [
            "online", "digital", "technology", "electronic",
        ],
        "department": "ICT / Network Support",
    },
    "Finance": {
        "description": "Fees, payments, scholarships, financial aid, and money matters",
        "strong_indicators": [
            "fee", "fees", "school fees", "tuition", "tuition fees",
            "payment", "pay", "paid", "paying", "invoice", "receipt",
            "scholarship", "bursary", "financial aid", "sponsorship", "grant", "funding",
            "money", "cash", "fund", "finance", "financial",
            "expensive", "cost", "price", "charge", "overcharge",
            "refund", "reimburse", "reimbursement",
            "loan", "debt", "owe", "outstanding",
            "bank", "transfer", "mobile money", "momo",
            "afford", "affordable", "unaffordable",
            "budget", "financial", "economy",
            "finance office", "bursar", "accounts",
        ],
        "phrase_patterns": [
            r"(?:fee|fees|tuition)\s+(?:is\s+)?(?:too\s+high|expensive|unaffordable|increased)",
            r"(?:cannot|can't|unable\s+to)\s+(?:pay|afford)",
            r"(?:scholarship|bursary|financial\s+aid)\s+(?:is\s+)?(?:not\s+(?:given|approved|available)|delayed|rejected)",
            r"(?:overcharged|overcharge|extra\s+charge|hidden\s+fee)",
            r"(?:refund|reimbursement)\s+(?:is\s+)?(?:not\s+(?:given|received)|delayed|pending)",
            r"(?:payment|invoice|receipt)\s+(?:is\s+)?(?:not\s+(?:accepted|processed|recognized)|problem|issue)",
            r"(?:loan|debt)\s+(?:is\s+)?(?:not\s+(?:approved|available)|delayed|problem)",
            r"(?:payment|fee|tuition)\s+(?:I\s+)?(?:made|paid)\s+(?:has\s+)?(?:not\s+been\s+(?:reflected|processed|recognized|updated)|is\s+still\s+(?:not\s+(?:reflected|showing|updated)))",
            r"(?:bursary|scholarship|financial\s+aid)\s+(?:application|process)\s+(?:is\s+)?(?:too\s+(?:complicated|slow|difficult)|delayed|not\s+(?:working|processing|approved|accepted)|problem|issue)",
            r"(?:account|statement)\s+(?:is\s+)?(?:not\s+(?:updated|reflecting|showing|correct)|wrong|incorrect|has\s+(?:an?\s+)?(?:error|discrepancy|problem))",
        ],
        "weak_indicators": [
            "money", "cost", "price",
        ],
        "department": "Finance Office / Bursar",
    },
    "Safety": {
        "description": "Security, crime, violence, emergencies, and personal safety",
        "strong_indicators": [
            "security", "guard", "guards", "watchman", "patrol", "patrolling",
            "theft", "stolen", "steal", "robbery", "robbed", "mug", "mugged",
            "attack", "attacked", "assault", "assaulted", "violence", "violent",
            "fight", "fighting", "harass", "harassed", "harassment",
            "threat", "threatened", "threatening", "intimidate", "intimidated",
            "kidnap", "kidnapped", "kidnapping", "abduct", "abducted", "hostage",
            "weapon", "gun", "gunshot", "shooting", "knife", "machete", "cutlass",
            "stab", "stabbed", "stabbing", "rape", "raped", "sexual assault",
            "bomb", "explosion", "explode", "exploded",
            "intruder", "break-in", "burglary", "trespass", "trespassing",
            "danger", "dangerous", "unsafe", "hazard", "hazardous",
            "emergency", "crisis", "critical", "urgent",
            "cctv", "surveillance", "camera",
            "fire", "fire outbreak", "burning", "smoke",
            "accident", "crash", "injured", "injury", "blood",
            "panic", "alarm", "scream",
            "dark", "unlit", "poor lighting", "no lighting",
            "suspicious", "stranger", "stalker", "lurking",
            "vandalism", "vandalized", "destroy", "destruction",
            "abuse", "abusive", "bully", "bullying",
        ],
        "phrase_patterns": [
            r"(?:was|got|been)\s+(?:robbed|attacked|assaulted|harassed|threatened|stabbed|kidnapped|mugged|raped|abducted|beaten|molested)",
            r"(?:someone|a\s+student|student)\s+(?:was|got|been)\s+(?:robbed|attacked|assaulted|harassed|threatened|injured|killed|stabbed|kidnapped|mugged)",
            r"(?:phone|laptop|bag|wallet|money|valuables)\s+(?:was|got|been)\s+(?:stolen|snatched|taken|robbed|grabbed)",
            r"(?:no|not|inadequate|poor|insufficient)\s+(?:security|guard|patrol|lighting|cctv|camera|surveillance|protection)",
            r"(?:feeling|felt|feel)\s+(?:unsafe|insecure|afraid|scared|threatened|vulnerable|worried|anxious|frightened|alarmed)",
            r"(?:dark|unlit|poor\s+lighting|no\s+lighting|dim)\s+(?:area|street|path|corridor|parking|entrance|staircase|passage)",
            r"(?:suspicious|strange|unknown|unidentified)\s+(?:person|people|man|men|activity|behavior|movement|vehicle)",
            r"(?:fire|smoke|burning)\s+(?:outbreak|incident|emergency|in\s+(?:building|hostel|room|office|campus))",
            r"(?:cctv|surveillance|camera)\s+(?:is\s+)?(?:not\s+working|broken|faulty|missing|inadequate|insufficient|blind\s+spot)",
            r"(?:break[\s-]?in|burglary|theft|stealing|robbery)\s+(?:on\s+campus|in\s+(?:hostel|room|hall|dorm)|near\s+campus|at\s+(?:the\s+)?(?:hostel|campus|gate))",
            r"(?:intrud(?:er|ers|ing)|trespass(?:er|ers|ing)|stalk(?:er|ing))\s+(?:on\s+campus|in\s+(?:hostel|hall|dorm|room|building))",
            r"(?:laptop|phone|bag|wallet)\s+(?:was\s+)?(?:snatched|grabbed|stolen|taken)\s+(?:by|from)",
            r"(?:someone|person|thief|thieves|burglar|robber)\s+(?:broke?\s+into|entered|broke?\s+in\s+to|forced\s+(?:their|his|her)\s+way\s+into)\s+(?:my|the|our)\s+(?:room|hostel|hall|house|office|car)",
            r"(?:cctv|surveillance|security\s+camera|security\s+cameras)\s+(?:in|at|near)\s+(?:the\s+)?(?:parking\s+lot|parking|campus|hostel|gate|entrance|building|corridor|hallway)",
        ],
        "weak_indicators": [
            "safe", "protect", "campus", "night",
        ],
        "department": "Campus Security / Emergency Response",
    },
    "Maintenance": {
        "description": "Building repairs, plumbing, electrical, furniture, and general upkeep",
        "strong_indicators": [
            "repair", "repairs", "fix", "fixed", "fixing", "broken", "faulty",
            "maintenance", "maintain", "service", "servicing",
            "plumber", "plumbing", "pipe", "pipes", "tap", "taps",
            "electric", "electrical", "electrician", "wiring", "cable",
            "carpenter", "carpentry", "wood", "wooden",
            "mason", "masonry", "cement", "concrete", "brick",
            "paint", "painting", "painted", "peeling",
            "door", "window", "lock", "handle", "hinge",
            "roof", "ceiling", "floor", "wall", "tile", "tiles",
            "furniture", "furnish", "furnished", "chair", "table", "desk", "bed", "mattress",
            "fan", "air conditioner", "ac", "cooling", "ventilation",
            "elevator", "lift", "escalator",
            "generator", "power", "electricity", "light", "lighting",
            "water", "water supply", "pump", "borehole",
            "drain", "drainage", "sewage", "sewer", "blockage", "blocked",
            "leak", "leaking", "leakage", "drip", "dripping", "burst",
            "flood", "flooded", "flooding", "overflow", "waterlogged",
            "crack", "cracked", "cracking", "collapse", "collapsed", "collapsing",
            "deteriorate", "deteriorating", "deterioration", "run down", "worn out",
            "damage", "damaged", "destroy", "destroyed",
            "rust", "rusty", "corrode", "corroded",
            "mold", "mildew", "damp", "musty",
        ],
        "phrase_patterns": [
            r"(?:door|window|lock|handle|hinge)\s+(?:is\s+)?(?:broken|damaged|not\s+working|faulty|missing)",
            r"(?:roof|ceiling|wall|floor)\s+(?:is\s+)?(?:leaking|cracked|damaged|collapsing|broken)",
            r"(?:pipe|tap|water)\s+(?:is\s+)?(?:leaking|burst|broken|not\s+working|blocked)",
            r"(?:fan|ac|air\s+conditioner|light|electricity|power)\s+(?:is\s+)?(?:not\s+working|broken|faulty|out)",
            r"(?:elevator|lift)\s+(?:is\s+)?(?:not\s+working|broken|stuck|out\s+of\s+order)",
            r"(?:furniture|chair|table|desk|bed|mattress)\s+(?:is\s+)?(?:broken|damaged|missing|uncomfortable)",
            r"(?:drain|sewage|toilet|bathroom)\s+(?:is\s+)?(?:blocked|clogged|overflowing|leaking)",
            r"(?:needs?|requires?)\s+(?:repair|fix|maintenance|attention)",
            r"(?:still|yet)\s+(?:not\s+fixed|not\s+repaired|not\s+maintained)",
            r"(?:water|electricity|power|light|fan)\s+(?:is\s+)?(?:not\s+flowing|not\s+working|off|out|cut\s+off|disconnected)",
            r"(?:no|not)\s+(?:water|electricity|power|light|fan|hot\s+water)\s+(?:in|at|flowing|working)",
            r"(?:fan|light|ac|air\s+conditioner)\s+(?:in\s+(?:the\s+)?(?:lecture\s+hall|classroom|library|lab|lab\s+room|office|hostel|dorm|hall))\s+(?:is\s+)?(?:not\s+working|broken|faulty|out)",
            r"(?:water|pipe|tap|shower)\s+(?:in\s+(?:the\s+)?(?:bathroom|toilet|washroom|restroom|kitchen|hostel|dorm|hall|room))\s+(?:is\s+)?(?:not\s+working|leaking|broken|blocked|not\s+flowing)",
        ],
        "weak_indicators": [
            "building", "structure", "facility", "infrastructure",
        ],
        "department": "Maintenance / Facilities Office",
    },
    "Accommodation": {
        "description": "Hostel, dormitory, room allocation, and living conditions",
        "strong_indicators": [
            "hostel", "dorm", "dormitory", "hall", "residence", "accommodation",
            "room", "bedroom", "chamber", "suite",
            "bed", "mattress", "pillow", "bedding", "sheet", "blanket",
            "cupboard", "wardrobe", "closet", "shelf", "shelves",
            "kitchen", "cooking", "pantry",
            "bathroom", "toilet", "washroom", "restroom", "shower", "bath",
            "laundry", "washing", "drying",
            "neighbor", "roommate", "suite mate", "hall mate",
            "noise", "noisy", "loud", "disturbance", "disturb",
            "crowd", "crowded", "overcrowded", "overcrowding",
            "space", "spacious", "cramped", "tiny",
            "allocate", "allocation", "room allocation", "placement",
            "move", "moving", "transfer", "relocate",
            "hostel fee", "hall fee", "accommodation fee",
            "warden", "hall master", "hall mistress", "hall tutor",
            "common room", "TV room", "study room",
            "visitation", "visitor", "guest",
            "curfew", "lockout", "late",
            "clean", "cleaning", "dirty", "hygiene", "sanitation",
            "pest", "pests", "cockroach", "roach", "rat", "rodent", "mouse", "mice",
            "mosquito", "insect", "bug", "infestation",
        ],
        "phrase_patterns": [
            r"(?:hostel|dorm|hall|room)\s+(?:is\s+)?(?:full|occupied|not\s+available|unavailable)",
            r"(?:no|not)\s+(?:bed|room|space|accommodation|hostel)",
            r"(?:room|hostel|hall)\s+(?:is\s+)?(?:too\s+small|cramped|overcrowded|noisy|dirty)",
            r"(?:bathroom|toilet|shower)\s+(?:is\s+)?(?:dirty|not\s+clean|blocked|not\s+working)",
            r"(?:neighbor|roommate|mate)\s+(?:is\s+)?(?:noisy|loud|disturbing|annoying|problem)",
            r"(?:cockroach|rat|rodent|roach|mosquito|pest)\s+(?:infestation|problem|in\s+(?:room|hostel|hall))",
            r"(?:room|hostel)\s+(?:allocation|placement)\s+(?:is\s+)?(?:unfair|delayed|problem|not\s+done)",
            r"(?:warden|hall\s+master|hall\s+mistress)\s+(?:is\s+)?(?:not\s+(?:available|helpful|responsive)|strict|harsh)",
        ],
        "weak_indicators": [
            "live", "living", "stay", "sleep",
        ],
        "department": "Hall Management / Accommodation Office",
    },
    "Staff": {
        "description": "Staff behavior, conduct, competence, and interactions",
        "strong_indicators": [
            "staff", "employee", "worker", "personnel",
            "lecturer", "professor", "teacher", "instructor", "tutor",
            "administrator", "admin", "officer",
            "secretary", "receptionist", "clerk",
            "cleaner", "janitor", "housekeeping",
            "security guard", "watchman", "porter",
            "caterer", "cook", "chef",
            "conduct", "behavior", "attitude", "character",
            "rude", "disrespectful", "insult", "insulting", "abusive",
            "unprofessional", "incompetent", "unqualified",
            "unhelpful", "unresponsive", "ignoring", "neglect",
            "harsh", "strict", "harassment", "intimidation",
            "corrupt", "corruption", "bribe", "bribery",
            "late", "absent", "absenteeism", "unpunctual",
            "lazy", "careless", "negligent",
            "discrimination", "discriminate", "bias", "biased",
            "favoritism", "nepotism", "victimization",
            "sexual harassment", "inappropriate", "misconduct",
            "complaint", "grievance",
        ],
        "phrase_patterns": [
            r"(?:lecturer|teacher|professor|staff|officer|employee|worker)\s+(?:is\s+)?(?:rude|disrespectful|unprofessional|incompetent|unhelpful|harsh|abusive|insulting|lazy|careless|negligent|unresponsive|hostile|arrogant|dictatorial)",
            r"(?:staff|employee|worker|personnel|lecturer|teacher|professor)\s+(?:is\s+)?(?:not\s+(?:helpful|responsive|professional|competent|qualified|available|present|coming))",
            r"(?:was|been)\s+(?:insulted|disrespected|abused|harassed|victimized|discriminated|intimidated|threatened)\s+by\s+(?:a\s+)?(?:lecturer|teacher|staff|officer|security|guard|professor|worker)",
            r"(?:bribe|bribery|corruption|corrupt)\s+(?:by|from|involving|demanded\s+by)\s+(?:staff|lecturer|officer|employee|worker|teacher|professor)",
            r"(?:favoritism|nepotism|bias|discrimination)\s+(?:by|from|in)\s+(?:staff|lecturer|teacher|professor|officer|department|management)",
            r"(?:sexual\s+harassment|inappropriate\s+(?:behavior|conduct|touching|comment|remark)|misconduct)\s+(?:by|from)\s+(?:lecturer|staff|officer|teacher|professor|employee|worker)",
            r"(?:lecturer|teacher|professor|staff|officer)\s+(?:is\s+)?(?:always\s+late|always\s+absent|never\s+comes|doesn't\s+come|never\s+available|never\s+teaching)",
            r"(?:security\s+guard|watchman|porter)\s+(?:is\s+)?(?:rude|disrespectful|abusive|harsh|unhelpful|lazy|careless|negligent|absent|not\s+(?:present|at\s+(?:his|their|the)\+post))",
            r"(?:cleaner|janitor|housekeeping)\s+(?:is\s+)?(?:lazy|careless|negligent|not\s+(?:cleaning|working|doing\s+(?:his|their)\s+job))",
            r"(?:caterer|cook|chef)\s+(?:is\s+)?(?:rude|unhygienic|unprofessional|careless|lazy|dirty)",
            r"(?:lecturer|teacher|professor)\s+(?:is\s+)?(?:very\s+rude|very\s+disrespectful|extremely\s+rude|always\s+insulting|never\s+teaching|doesn't\s+care|not\s+caring)",
            r"(?:lecturer|teacher|professor)\s+(?:makes?|making)\s+(?:inappropriate|sexual|offensive|disrespectful|rude|derogatory)\s+(?:comments?|remarks?|jokes?|gestures?)",
        ],
        "weak_indicators": [
            "person", "people", "worker",
        ],
        "department": "Human Resources / Administration",
    },
    "Administration": {
        "description": "Administrative processes, registry, records, and institutional bureaucracy",
        "strong_indicators": [
            "admin", "administration", "administrative",
            "registry", "registrar", "records",
            "office", "desk", "counter", "front desk", "help desk", "helpdesk",
            "secretary", "reception", "receptionist",
            "document", "documents", "certificate", "certificates",
            "transcript", "transcripts", "academic transcript",
            "letter", "letters", "reference letter", "recommendation letter",
            "verification", "verification letter", "confirmation",
            "admission", "admissions", "enrollment", "enrolment", "registration",
            "process", "processing", "procedure", "procedures",
            "bureaucracy", "bureaucratic", "red tape",
            "delay", "delayed", "delays", "pending", "waiting",
            "form", "forms", "application", "apply",
            "policy", "policies", "rule", "rules", "regulation", "regulations",
            "committee", "board", "council", "senate",
            "announcement", "notice", "circular",
            "meeting", "minutes", "agenda",
            "src", "student representative", "student union",
            "dean", "provice chancellor", "vice chancellor", "rector",
            "head of department", "hod", "coordinator",
        ],
        "phrase_patterns": [
            r"(?:registry|registrar|office)\s+(?:is\s+)?(?:slow|unhelpful|not\s+responsive|delayed|closed)",
            r"(?:transcript|certificate|document|letter)\s+(?:is\s+)?(?:not\s+(?:ready|available|processed|issued)|delayed|pending)",
            r"(?:admission|enrollment|registration)\s+(?:is\s+)?(?:delayed|not\s+(?:processed|complete|done)|problem|issue)",
            r"(?:application|form|request)\s+(?:is\s+)?(?:not\s+(?:processed|approved|accepted)|delayed|pending|rejected)",
            r"(?:long|too\s+long|endless)\s+(?:process|procedure|delay|queue|waiting)",
            r"(?:no|not)\s+(?:response|reply|feedback|update|information)\s+(?:from|after)",
            r"(?:bureaucracy|red\s+tape|too\s+many\s+(?:steps|processes|procedures|forms))",
            r"(?:online\s+registration|registration\s+system|student\s+portal|online\s+system)\s+(?:is\s+)?(?:efficient|good|excellent|working\s+well|reliable|bad|poor|terrible|slow|broken)",
            r"(?:certificate|transcript|document)\s+(?:still\s+)?(?:not\s+ready|not\s+available|not\s+processed|delayed|pending|missing)",
            r"(?:front\s+desk|help\s+desk|reception)\s+(?:is\s+)?(?:unhelpful|rude|not\s+helpful|slow|unresponsive|unprofessional)",
        ],
        "weak_indicators": [
            "system", "process", "official",
        ],
        "department": "Administration / Registry",
    },
    "Transport": {
        "description": "Buses, shuttles, parking, and campus transportation",
        "strong_indicators": [
            "bus", "buses", "shuttle", "vehicle", "vehicles",
            "transport", "transportation", "transit",
            "driver", "drivers", "conductor",
            "route", "routes", "stop", "station",
            "parking", "park", "car park", "parking lot", "parking space",
            "trotro", "taxi", "uber", "bolt",
            "fare", "fares", "ticket", "tickets",
            "schedule", "timetable", "departure", "arrival",
            "late", "delay", "delayed", "punctual", "punctuality",
            "accident", "crash", "breakdown",
            "road", "roads", "street", "highway",
            "traffic", "congestion", "jam",
            "fuel", "petrol", "diesel",
            "motorcycle", "bike", "bicycle",
            "walk", "walking", "pedestrian",
            "garage", "mechanic", "repair",
        ],
        "phrase_patterns": [
            r"(?:bus|shuttle|transport)\s+(?:is\s+)?(?:late|delayed|not\s+coming|not\s+available|full|cancelled|broken|unreliable|terrible|awful)",
            r"(?:no|not)\s+(?:bus|shuttle|transport)\s+(?:to|from|for|available|running|service)",
            r"(?:driver|conductor)\s+(?:is\s+)?(?:rude|reckless|speeding|drunk|unprofessional|unhelpful|careless|dangerous)",
            r"(?:parking|car\s+park)\s+(?:is\s+)?(?:full|not\s+available|too\s+small|too\s+far|expensive|unsafe|inadequate)",
            r"(?:fare|ticket|price)\s+(?:is\s+)?(?:too\s+high|expensive|increased|unaffordable|costly)",
            r"(?:route|schedule|timetable)\s+(?:is\s+)?(?:not\s+(?:convenient|available|reliable)|changed|unclear|inadequate|poor)",
            r"(?:road|street)\s+(?:is\s+)?(?:bad|dangerous|not\s+safe|flooded|pothole|dark|unlit|poor)",
            r"(?:bus\s+service|shuttle\s+service|transport\s+service)\s+(?:is\s+)?(?:good|excellent|reliable|efficient|bad|poor|terrible|unreliable|not\s+available|not\s+running)",
        ],
        "weak_indicators": [
            "travel", "commute", "journey",
        ],
        "department": "Transport Services / Security",
    },
    "Catering": {
        "description": "Food, dining, canteen, and catering services",
        "strong_indicators": [
            "food", "meal", "meals", "cuisine",
            "canteen", "cafeteria", "dining hall", "dining", "restaurant",
            "kitchen", "cook", "cooking", "chef", "caterer",
            "menu", "dish", "dishes",
            "breakfast", "lunch", "dinner", "supper",
            "rice", "banku", "jollof", "waakye", "fufu", "kenkey",
            "meat", "chicken", "fish", "egg", "eggs",
            "vegetable", "vegetables", "fruit", "fruits",
            "water", "drink", "drinks", "beverage",
            "hungry", "hunger", "starving", "starvation",
            "taste", "tasty", "delicious", "flavor", "bland",
            "fresh", "stale", "rotten", "spoiled", "expired", "contaminated",
            "hygiene", "hygienic", "unhygienic", "dirty", "clean",
            "portion", "size", "quantity", "amount",
            "price", "cost", "expensive", "overpriced", "afford",
            "fly", "flies", "worm", "worms", "hair", "stone", "stones",
            "food poisoning", "stomach", "diarrhea", "cholera",
            "vendor", "vendors", "food vendor",
            "serving", "serve", "served",
            "queue", "line", "waiting",
        ],
        "phrase_patterns": [
            r"(?:food|meal|rice|dish)\s+(?:is\s)?(?:cold|stale|rotten|spoiled|expired|contaminated|bland|tasteless|undercooked|overcooked|unhygienic|dirty|terrible|horrible|disgusting|awful)",
            r"(?:canteen|cafeteria|kitchen|dining)\s+(?:is\s)?(?:dirty|unhygienic|unclean|filthy|smelly)",
            r"(?:found|saw|there\s+(?:is|are))\s+(?:fly|flies|worm|worms|hair|stone|insect|cockroach)\s+(?:in|on)\s+(?:food|meal|rice|dish|soup)",
            r"(?:portion|size|quantity)\s+(?:is\s)?(?:too\s+small|small|little|insufficient|inadequate)",
            r"(?:food|meal|price|canteen)\s+(?:is\s)?(?:too\s+expensive|expensive|overpriced|unaffordable|costly)",
            r"(?:menu|food)\s+(?:is\s)?(?:not\s+varied|same|repetitive|boring|limited|monotonous)",
            r"(?:food\s+poisoning|stomach\s+(?:pain|ache|problem)|diarrhea|vomiting)\s+(?:after|from)\s+(?:eating|food|canteen|meal)",
            r"(?:no|not)\s+(?:food|meal|breakfast|lunch|dinner)\s+(?:available|served|provided)",
            r"(?:service|staff)\s+(?:at\s+(?:the\s+)?(?:canteen|cafeteria|dining|food))\s+(?:is\s+)?(?:slow|terrible|poor|bad|rude|unfriendly|unhelpful)",
        ],
        "weak_indicators": [
            "eat", "eating", "dinner", "lunch",
        ],
        "department": "Catering Services / Food Services",
    },
    "Library": {
        "description": "Library services, resources, study spaces, and academic materials",
        "strong_indicators": [
            "library", "librarian", "libraries",
            "book", "books", "textbook", "textbooks",
            "journal", "journals", "article", "articles",
            "e-library", "electronic library", "digital library",
            "database", "databases", "jstor", "pubmed", "google scholar",
            "catalog", "catalogue", "search",
            "borrow", "borrowing", "loan", "lend", "return",
            "reference", "references", "citation", "citations",
            "study area", "study room", "study space", "reading room",
            "reading", "read", "quiet", "silence", "silent",
            "resource", "resources", "material", "materials",
            "periodical", "periodicals", "magazine", "magazines",
            "newspaper", "newspapers",
            "archive", "archives", "special collection",
            "thesis", "dissertation", "project",
            "photocopy", "photocopying", "photostat",
            "library card", "library fine", "overdue",
            "opening hours", "closing hours", "library hours",
            "seat", "seating", "space", "table",
            "computer", "internet", "wifi",
            "air conditioning", "ac", "fan", "lighting",
        ],
        "phrase_patterns": [
            r"(?:library|e-library)\s+(?:is\s+)?(?:closed|not\s+open|too\s+crowded|too\s+noisy|too\s+hot|uncomfortable)",
            r"(?:book|textbook|journal|article|resource)\s+(?:is\s+)?(?:not\s+(?:available|found|in\s+(?:stock|library))|missing|outdated|old|insufficient)",
            r"(?:no|not)\s+(?:book|textbook|journal|resource|material|computer|internet|wifi|seat|space)\s+(?:in|at|available\s+in)\s+(?:the\s+)?library",
            r"(?:library|study\s+area|reading\s+room)\s+(?:is\s+)?(?:too\s+small|cramped|insufficient|inadequate|uncomfortable)",
            r"(?:opening|closing|library)\s+(?:hours|time)\s+(?:is\s+)?(?:not\s+(?:convenient|sufficient|adequate)|too\s+short|too\s+early|too\s+late)",
            r"(?:cannot|can't|unable\s+to)\s+(?:borrow|access|find|locate|search|download|photocopy)\s+(?:book|journal|article|resource|material)",
            r"(?:librarian|library\s+staff)\s+(?:is\s+)?(?:unhelpful|rude|unavailable|not\s+(?:present|around|helpful))",
            r"(?:librarian|library\s+staff)\s+(?:is\s+)?(?:never\s+(?:at|in|available|present)|always\s+(?:absent|away|missing|unavailable))",
        ],
        "weak_indicators": [
            "study", "research", "read", "book",
        ],
        "department": "Library Services",
    },
    "Student Affairs": {
        "description": "Student life, clubs, sports, health services, counseling, and welfare",
        "strong_indicators": [
            "student affairs", "student life", "student welfare",
            "club", "clubs", "society", "societies", "association", "associations",
            "sports", "sport", "game", "games", "athletics",
            "football", "basketball", "volleyball", "tennis", "badminton",
            "gym", "gymnasium", "fitness", "exercise",
            "competition", "tournament", "league", "match",
            "event", "events", "program", "programme", "activity", "activities",
            "cultural", "culture", "entertainment", "recreation",
            "health", "health center", "clinic", "hospital", "medical",
            "doctor", "nurse", "physician", "medical",
            "counseling", "counselling", "counselor", "therapist", "therapy",
            "mental health", "stress", "anxiety", "depression", "wellness", "wellbeing",
            "disability", "special needs", "accommodation",
            "orientation", "induction", "welcome",
            "graduation", "convocation", "ceremony",
            "id card", "student id", "identification",
            "welfare", "welfare fund", "emergency fund",
            "peer", "mentor", "mentoring", "buddy",
            "leadership", "training", "development",
            "volunteer", "volunteering", "community service",
            "alumni", "former student",
        ],
        "phrase_patterns": [
            r"(?:club|society|association|sports|team)\s+(?:is\s+)?(?:not\s+(?:active|functioning|supported|funded|recognized)|inactive|dead|disbanded)",
            r"(?:no|not)\s+(?:sports|gym|fitness|recreation|entertainment|cultural|event|activity|program)",
            r"(?:health\s+center|clinic|hospital|medical)\s+(?:is\s+)?(?:closed|not\s+(?:open|available|functioning|helpful)|understaffed|under-equipped)",
            r"(?:counseling|counselling|therapy|mental\s+health)\s+(?:is\s+)?(?:not\s+(?:available|accessible|helpful)|insufficient|inadequate)",
            r"(?:doctor|nurse|medical)\s+(?:is\s+)?(?:not\s+(?:available|present|on\s+duty|helpful|responsive))",
            r"(?:id\s+card|student\s+id|identification)\s+(?:is\s+)?(?:not\s+(?:issued|ready|available|processed)|lost|stolen|damaged)",
            r"(?:event|competition|tournament|match|program)\s+(?:is\s+)?(?:cancelled|canceled|postponed|not\s+(?:organized|held|supported))",
            r"(?:orientation|induction|welcome)\s+(?:is\s+)?(?:not\s+(?:organized|held|done)|poor|inadequate|insufficient)",
            r"(?:gym|sports\s+complex|fitness\s+center|athletic)\s+(?:equipment|facility|facilities|field|court)\s+(?:is\s+)?(?:broken|damaged|outdated|old|inadequate|insufficient|poor|dangerous|unsafe|not\s+working)",
            r"(?:sports|athletic|recreational)\s+(?:equipment|facility|facilities|field|court|ground)\s+(?:is\s+)?(?:broken|damaged|outdated|old|inadequate|insufficient|poor|dangerous|unsafe|not\s+working|needs?\s+(?:repair|replacement|maintenance|upgrade))",
        ],
        "weak_indicators": [
            "student", "activity", "program",
        ],
        "department": "Student Affairs / Dean of Students",
    },
}


# ==================== HELPER FUNCTIONS ====================

def _normalize_text(text: str) -> str:
    """Normalize text for analysis."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _detect_negation(text: str, keyword: str) -> bool:
    """Check if a keyword is negated in context."""
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


def _has_resolution_language(text: str) -> bool:
    """Check if the text contains resolution/recovery language."""
    resolution_patterns = [
        r"\b(?:fixed\s+now|working\s+again|back\s+to\s+normal|problem\s+solved|issue\s+resolved)\b",
        r"\b(?:has\s+been\s+fixed|have\s+been\s+fixed|has\s+been\s+resolved|have\s+been\s+resolved)\b",
        r"\b(?:finally\s+fixed|finally\s+resolved|finally\s+solved)\b",
        r"\b(?:no\s+longer|no\s+more)\s+(?:a\s+)?(?:problem|issue|concern|leak|broken)",
        r"\b(?:used\s+to|before|previously|earlier|last\s+semester|last\s+week|last\s+month|last\s+year)\s+.*\s+(?:but\s+now|but\s+currently|but\s+these\s+days|has\s+improved|has\s+gotten\s+better|is\s+better|is\s+fixed|is\s+solved)\b",
        r"\b(?:improved|better|resolved|sorted\s+out|dealt\s+with|addressed|fixed|repaired|upgraded)\b",
        r"\b(?:maintenance\s+fixed|maintenance\s+repaired|someone\s+fixed|they\s+fixed|it\s+was\s+fixed)\b",
        r"\b(?:was\s+a\s+.*\s+but\s+(?:now|currently|this\s+semester))\b",
    ]
    text_lower = text.lower()
    for pattern in resolution_patterns:
        if re.search(pattern, text_lower):
            return True
    return False


def _has_contrast_structure(text: str) -> bool:
    """Check if the text has a contrast structure (positive + negative)."""
    contrast_markers = [
        "but", "however", "although", "though", "yet", "except", "while", "whereas",
        "nevertheless", "on the other hand", "on the contrary", "in contrast", "despite",
        "unfortunately", "apart from", "other than", "aside from", "even though",
        "even if", "regardless", "nonetheless", "still", "albeit",
    ]
    text_lower = text.lower()
    return any(marker in text_lower for marker in contrast_markers)


def _split_clauses(text: str) -> List[str]:
    """Split text into clauses for mixed sentiment analysis."""
    # Split on common clause separators
    clauses = re.split(r'[;,]|\bbut\b|\bhowever\b|\balthough\b|\bthough\b|\byet\b', text)
    return [c.strip() for c in clauses if c.strip()]


def _analyze_clause_sentiment(clause: str) -> str:
    """Analyze sentiment of a single clause."""
    from sentiment_analyzer import _analyze_sentiment_internal
    sentiment, _, _ = _analyze_sentiment_internal(clause)
    return sentiment


# ==================== CATEGORY CLASSIFICATION ====================

def classify_categories(text: str) -> List[CategoryMatch]:
    """
    Classify feedback into one or more categories using semantic analysis.
    
    Returns a list of CategoryMatch objects sorted by confidence (highest first).
    """
    if not text or not text.strip():
        return [CategoryMatch(name="Other", confidence=0.0, evidence=[], is_primary=True)]
    
    normalized = _normalize_text(text)
    scores: Dict[str, Dict[str, Any]] = {}
    
    for cat_name, cat_def in CATEGORY_DEFINITIONS.items():
        evidence: List[str] = []
        score = 0.0
        
        # Check phrase patterns (strongest signal) - these are category-specific
        phrase_matched = False
        for pattern in cat_def.get("phrase_patterns", []):
            matches = re.findall(pattern, normalized, re.IGNORECASE)
            if matches:
                evidence.extend(matches if isinstance(matches[0], str) else [pattern])
                score += 4.0 * len(matches)  # Increased weight for phrase patterns
                phrase_matched = True
        
        # Check strong indicators
        for indicator in cat_def.get("strong_indicators", []):
            if _detect_negation(normalized, indicator):
                continue
            # Use word boundary matching for all indicators to avoid
            # matching inside unrelated words (e.g., "class" in "classroom")
            pattern = r"(?<!\w)" + re.escape(indicator) + r"(?!\w)"
            if re.search(pattern, normalized):
                evidence.append(indicator)
                score += 1.0
        
        # Check weak indicators (only count if we already have some evidence)
        if score > 0:
            for indicator in cat_def.get("weak_indicators", []):
                if _detect_negation(normalized, indicator):
                    continue
                if len(indicator) <= 3:
                    pattern = r"(?<!\w)" + re.escape(indicator) + r"(?!\w)"
                    if re.search(pattern, normalized):
                        evidence.append(indicator)
                        score += 0.3
                else:
                    if indicator in normalized:
                        evidence.append(indicator)
                        score += 0.3
        
        if score > 0:
            # Normalize score to 0-1 range (cap at reasonable max)
            confidence = min(1.0, score / 10.0)  # Adjusted denominator
            scores[cat_name] = {
                "score": score,
                "confidence": confidence,
                "evidence": list(set(evidence)),  # deduplicate
            }
    
    if not scores:
        return [CategoryMatch(name="Other", confidence=0.0, evidence=[], is_primary=True)]
    
    # Sort by score descending
    sorted_cats = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)
    
    # Create CategoryMatch objects
    results = []
    for i, (cat_name, data) in enumerate(sorted_cats):
        match = CategoryMatch(
            name=cat_name,
            confidence=round(data["confidence"], 3),
            evidence=data["evidence"],
            is_primary=(i == 0),
        )
        results.append(match)
    
    # Mark primary
    if results:
        results[0].is_primary = True
    
    return results


# ==================== SENTIMENT ANALYSIS ====================

def analyze_sentiment_type(text: str, sentiment_label: str, sentiment_score: float) -> str:
    """
    Determine the detailed sentiment type: positive, negative, neutral, mixed, or unclear.
    """
    if not text or not text.strip():
        return "unclear"
    
    # Very short responses are unclear
    words = text.split()
    if len(words) <= 2:
        # Check if it's a clear positive/negative word
        clear_positive = ["excellent", "amazing", "wonderful", "fantastic", "great", "good", "thank", "thanks", "awesome", "perfect", "love", "best", "superb", "outstanding"]
        clear_negative = ["terrible", "horrible", "awful", "worst", "bad", "hate", "disgusting", "pathetic", "useless", "broken"]
        text_lower = text.lower().strip()
        if text_lower in clear_positive:
            return "positive"
        if text_lower in clear_negative:
            return "negative"
        return "unclear"
    
    # If the base sentiment is unclear (very close to 0 with low confidence)
    if abs(sentiment_score) < 0.05:
        # Check if there's any meaningful content
        if len(words) < 3:
            return "unclear"
        # Check for mixed sentiment even when overall score is near zero
        if _has_contrast_structure(text):
            clauses = _split_clauses(text)
            if len(clauses) >= 2:
                clause_sentiments = [_analyze_clause_sentiment(c) for c in clauses]
                has_positive = "Positive" in clause_sentiments
                has_negative = "Negative" in clause_sentiments
                if has_positive and has_negative:
                    return "mixed"
        return "neutral"
    
    # Check for mixed sentiment
    if _has_contrast_structure(text):
        clauses = _split_clauses(text)
        if len(clauses) >= 2:
            clause_sentiments = [_analyze_clause_sentiment(c) for c in clauses]
            has_positive = "Positive" in clause_sentiments
            has_negative = "Negative" in clause_sentiments
            if has_positive and has_negative:
                return "mixed"
    
    # Check for resolution language (past negative, now positive)
    if _has_resolution_language(text) and sentiment_score > 0:
        return "mixed"
    
    # Also check for resolution language with near-neutral score
    if _has_resolution_language(text) and abs(sentiment_score) < 0.2:
        return "mixed"
    
    # Map base sentiment
    if sentiment_score >= 0.05:
        return "positive"
    elif sentiment_score <= -0.05:
        return "negative"
    else:
        return "neutral"


# ==================== URGENCY DETERMINATION ====================

def determine_urgency(urgency_score: int, sentiment_type: str, categories: List[CategoryMatch], text: str = "") -> str:
    """Determine the urgency level."""
    # Critical safety terms always result in critical urgency
    # Only the most severe incidents are critical
    critical_terms = ["shooting", "gunshot", "gunshots", "kidnapped", "kidnapping", "hostage", "hostages", "bomb", "explosion", "exploded", "raped", "stabbing", "armed attack", "mass shooting", "fire outbreak", "building fire", "laboratory fire"]
    text_lower = text.lower() if text else ""
    has_critical_safety = any(term in text_lower for term in critical_terms)
    
    if has_critical_safety:
        return "critical"
    
    # Check for persistent issues (e.g., "down for a week", "not working for days")
    persistent_patterns = [
        r"(?:down|not\s+working|broken|offline|unavailable)\s+for\s+(?:a\s+)?(?:week|weeks|month|months|days|days)",
        r"(?:still|yet)\s+(?:not\s+(?:working|fixed|resolved|repaired)|down|broken|offline)",
        r"(?:has\s+been|have\s+been)\s+(?:down|not\s+working|broken|offline|unavailable)\s+for",
    ]
    is_persistent = any(re.search(p, text_lower) for p in persistent_patterns)
    
    # Safety issues are always at least high urgency
    if categories and categories[0].name == "Safety":
        # Only the most severe safety incidents are critical
        severe_terms = ["shooting", "gunshot", "gunshots", "kidnapped", "kidnapping", "hostage", "hostages", "bomb", "explosion", "exploded", "raped", "stabbing", "armed attack", "mass shooting", "fire outbreak", "building fire"]
        has_severe = any(term in text_lower for term in severe_terms)
        if has_severe:
            return "critical"
        if urgency_score >= 4:
            # Check if it's a less severe injury/robbery
            less_severe = ["injured", "injury", "robbed", "robbery", "theft", "stolen"]
            if any(term in text_lower for term in less_severe):
                return "high"
            return "critical"
        # Positive/neutral safety feedback should not be high urgency
        if sentiment_type in ("positive", "neutral"):
            return "low"
        return "high"
    
    # Persistent non-safety issues should be at least medium urgency
    if is_persistent and urgency_score < 2:
        urgency_score = 2
    
    if urgency_score >= 4:
        return "critical"
    elif urgency_score >= 3:
        return "high"
    elif urgency_score >= 2:
        return "medium"
    else:
        return "low"


# ==================== STUDENT RECOMMENDATION GENERATION ====================

def generate_student_recommendation(
    text: str,
    category: str,
    sentiment_type: str,
    urgency: str,
    categories: List[CategoryMatch],
    confidence: float,
) -> StudentRecommendation:
    """Generate a simple, relevant recommendation for the student."""
    
    # If confidence is too low, return a fallback (uses same threshold as generate_recommendation)
    if confidence < FALLBACK_CONFIDENCE_THRESHOLD:
        return StudentRecommendation(
            summary="We received your feedback but need more details to provide specific guidance.",
            immediate_action="Please provide more details about your concern, including specific location, time, and what you've already tried.",
            who_to_contact="SRC Secretariat",
            expected_timeline="Once more details are provided",
        )
    
    # Get category-specific recommendation generators
    generators = {
        "Academics": _student_rec_academics,
        "ICT": _student_rec_ict,
        "Finance": _student_rec_finance,
        "Safety": _student_rec_safety,
        "Maintenance": _student_rec_maintenance,
        "Accommodation": _student_rec_accommodation,
        "Staff": _student_rec_staff,
        "Administration": _student_rec_administration,
        "Transport": _student_rec_transport,
        "Catering": _student_rec_catering,
        "Library": _student_rec_library,
        "Student Affairs": _student_rec_student_affairs,
    }
    
    generator = generators.get(category, _student_rec_generic)
    return generator(text, sentiment_type, urgency, categories, confidence)


def _student_rec_academics(text, sentiment_type, urgency, categories, confidence):
    if sentiment_type == "positive":
        return StudentRecommendation(
            summary="Thank you for your positive feedback about academic matters.",
            immediate_action="Consider sharing your positive experience with peers or writing a testimonial for the department.",
            who_to_contact="Your Department or Academic Advisor",
            expected_timeline="N/A",
            additional_tips=["Your feedback helps recognize good teaching practices."],
        )
    elif sentiment_type == "mixed":
        return StudentRecommendation(
            summary="We understand you have both positive and negative experiences with academic matters.",
            immediate_action="Please speak with your course representative or academic advisor to discuss the specific concerns while also acknowledging what's working well.",
            who_to_contact="Course Representative / Academic Advisor",
            expected_timeline="3-5 days",
        )
    else:
        return StudentRecommendation(
            summary="We understand you're experiencing an academic concern.",
            immediate_action="Contact your course representative or academic advisor. If it's about grades, you can request a re-check within the official period. Document specific incidents with dates.",
            who_to_contact="Department Head / Academic Affairs Office",
            expected_timeline="3-7 days",
            additional_tips=[
                "Keep records of specific incidents (dates, course names, what happened)",
                "If it's a grade concern, check the official re-check/appeal deadline",
                "Your course rep can raise collective issues on your behalf",
            ],
        )


def _student_rec_ict(text, sentiment_type, urgency, categories, confidence):
    if sentiment_type == "positive":
        return StudentRecommendation(
            summary="Thank you for your positive feedback about ICT services.",
            immediate_action="Share your positive experience and any tips with fellow students.",
            who_to_contact="ICT Helpdesk",
            expected_timeline="N/A",
        )
    elif sentiment_type == "mixed":
        return StudentRecommendation(
            summary="We understand you have mixed experiences with ICT services.",
            immediate_action="Report the specific issues you're facing to the ICT helpdesk, mentioning what works well and what doesn't.",
            who_to_contact="ICT Helpdesk",
            expected_timeline="1-3 days",
        )
    else:
        return StudentRecommendation(
            summary="We understand you're experiencing an ICT issue.",
            immediate_action="Try basic troubleshooting: restart your device, clear browser cache, try a different browser. If the issue persists, report it with the exact error message, time, and location.",
            who_to_contact="ICT Helpdesk / Network Support",
            expected_timeline="1-3 days",
            additional_tips=[
                "Note the exact error message and when it occurs",
                "Try connecting from a different location to isolate the issue",
                "For portal issues, check if your password has expired",
            ],
        )


def _student_rec_finance(text, sentiment_type, urgency, categories, confidence):
    if sentiment_type == "positive":
        return StudentRecommendation(
            summary="Thank you for your positive feedback about financial services.",
            immediate_action="Share your experience with fellow students who may benefit.",
            who_to_contact="Finance Office",
            expected_timeline="N/A",
        )
    elif sentiment_type == "mixed":
        return StudentRecommendation(
            summary="We understand you have mixed experiences with financial matters.",
            immediate_action="Visit the Finance Office to discuss your specific concerns. Bring your student ID and any relevant documents.",
            who_to_contact="Finance Office / Bursar",
            expected_timeline="3-5 days",
        )
    else:
        return StudentRecommendation(
            summary="We understand you have a financial concern.",
            immediate_action="Visit the Finance Office with your student ID and relevant documents. If it's about fees, request a statement of account. If it's about scholarships, contact the Scholarships Office.",
            who_to_contact="Finance Office / Scholarships Office",
            expected_timeline="3-7 days",
            additional_tips=[
                "Bring your student ID and any payment receipts",
                "For scholarship inquiries, check application deadlines",
                "Request a formal statement of account for fee disputes",
            ],
        )


def _student_rec_safety(text, sentiment_type, urgency, categories, confidence):
    if urgency == "critical":
        return StudentRecommendation(
            summary="⚠️ This is a safety emergency. Your safety is the top priority.",
            immediate_action="Contact Campus Security IMMEDIATELY. If in danger, call emergency services. Move to a safe location. Do not confront the situation alone.",
            who_to_contact="Campus Security / Emergency Services",
            expected_timeline="Immediate",
            additional_tips=[
                "Save the campus security number in your phone",
                "Stay with others, avoid isolated areas",
                "Report the incident with as much detail as possible",
            ],
        )
    elif sentiment_type == "positive":
        return StudentRecommendation(
            summary="Thank you for your positive feedback about campus safety.",
            immediate_action="Continue to stay vigilant and report any concerns promptly.",
            who_to_contact="Campus Security",
            expected_timeline="N/A",
        )
    else:
        return StudentRecommendation(
            summary="We take your safety concern seriously.",
            immediate_action="Report the concern to Campus Security immediately. If it's about a specific incident, file a formal report. Avoid the area if it's an ongoing concern.",
            who_to_contact="Campus Security",
            expected_timeline="1-3 days",
            additional_tips=[
                "Always report safety incidents, even minor ones",
                "Walk in groups, especially at night",
                "Know the location of emergency phones on campus",
                "Save campus security contact in your phone",
            ],
        )


def _student_rec_maintenance(text, sentiment_type, urgency, categories, confidence):
    if sentiment_type == "positive":
        return StudentRecommendation(
            summary="Thank you for acknowledging maintenance improvements.",
            immediate_action="Continue to report any new issues promptly.",
            who_to_contact="Maintenance Office",
            expected_timeline="N/A",
        )
    elif sentiment_type == "mixed":
        return StudentRecommendation(
            summary="We understand some maintenance issues have been resolved while others remain.",
            immediate_action="Report the remaining issues to the Maintenance Office with specific locations and details.",
            who_to_contact="Maintenance Office / Facilities",
            expected_timeline="3-7 days",
        )
    else:
        return StudentRecommendation(
            summary="We understand you have a maintenance concern.",
            immediate_action="Report the specific issue to the Maintenance Office with the exact location (building, room number, floor). If it's urgent (water leak, electrical), report it immediately.",
            who_to_contact="Maintenance Office / Facilities",
            expected_timeline="2-5 days",
            additional_tips=[
                "Provide exact location: building, floor, room number",
                "For urgent issues (electrical, flooding), report immediately",
                "Take a photo of the problem if possible",
            ],
        )


def _student_rec_accommodation(text, sentiment_type, urgency, categories, confidence):
    if sentiment_type == "positive":
        return StudentRecommendation(
            summary="Thank you for your positive feedback about accommodation.",
            immediate_action="Share your positive experience with Hall Management.",
            who_to_contact="Hall Management",
            expected_timeline="N/A",
        )
    elif sentiment_type == "mixed":
        return StudentRecommendation(
            summary="We understand you have mixed experiences with your accommodation.",
            immediate_action="Speak with your Hall Warden or Hall Management about the specific concerns while acknowledging what's working.",
            who_to_contact="Hall Warden / Hall Management",
            expected_timeline="3-5 days",
        )
    else:
        return StudentRecommendation(
            summary="We understand you have an accommodation concern.",
            immediate_action="Report the issue to your Hall Warden or Hall Management. For maintenance issues in your room, submit a maintenance request. For roommate issues, speak with your Hall Tutor.",
            who_to_contact="Hall Warden / Hall Management",
            expected_timeline="2-5 days",
            additional_tips=[
                "Report issues early before they worsen",
                "For maintenance, provide your room number and specific problem",
                "For roommate conflicts, involve your Hall Tutor as mediator",
            ],
        )


def _student_rec_staff(text, sentiment_type, urgency, categories, confidence):
    if sentiment_type == "positive":
        return StudentRecommendation(
            summary="Thank you for recognizing good staff conduct.",
            immediate_action="Consider writing a commendation letter to the department head.",
            who_to_contact="Department Head / Administration",
            expected_timeline="N/A",
        )
    elif sentiment_type == "mixed":
        return StudentRecommendation(
            summary="We understand you have mixed experiences with staff.",
            immediate_action="Report specific concerns to the Department Head or Administration while also acknowledging positive interactions.",
            who_to_contact="Department Head / Administration",
            expected_timeline="5-7 days",
        )
    else:
        return StudentRecommendation(
            summary="We take concerns about staff conduct seriously.",
            immediate_action="Report the specific incident(s) to the Department Head or Administration. Include dates, times, and what was said or done. If it's about a lecturer, also inform the Dean of Faculty.",
            who_to_contact="Department Head / Dean of Faculty / Administration",
            expected_timeline="5-10 days",
            additional_tips=[
                "Document specific incidents with dates and details",
                "If possible, have a witness corroborate your account",
                "For serious misconduct, report directly to the Dean",
                "You have the right to file a formal complaint",
            ],
        )


def _student_rec_administration(text, sentiment_type, urgency, categories, confidence):
    if sentiment_type == "positive":
        return StudentRecommendation(
            summary="Thank you for your positive feedback about administrative services.",
            immediate_action="Share your positive experience with fellow students.",
            who_to_contact="Administration / Registry",
            expected_timeline="N/A",
        )
    elif sentiment_type == "mixed":
        return StudentRecommendation(
            summary="We understand you have mixed experiences with administrative processes.",
            immediate_action="Follow up on pending matters with the relevant office. Bring all required documents.",
            who_to_contact="Administration / Registry",
            expected_timeline="3-7 days",
        )
    else:
        return StudentRecommendation(
            summary="We understand you're experiencing an administrative issue.",
            immediate_action="Visit the relevant office with your student ID and all required documents. If your issue is not resolved within the stated timeframe, escalate to the Head of Department or the Dean.",
            who_to_contact="Registry / Administration / Head of Department",
            expected_timeline="3-10 days",
            additional_tips=[
                "Always bring your student ID",
                "Keep copies of all submitted documents",
                "Note the name of the officer you speak with",
                "Follow up if you don't hear back within the stated timeframe",
            ],
        )


def _student_rec_transport(text, sentiment_type, urgency, categories, confidence):
    if sentiment_type == "positive":
        return StudentRecommendation(
            summary="Thank you for your positive feedback about transport services.",
            immediate_action="Share your positive experience with the Transport Office.",
            who_to_contact="Transport Office",
            expected_timeline="N/A",
        )
    elif sentiment_type == "mixed":
        return StudentRecommendation(
            summary="We understand you have mixed experiences with transport services.",
            immediate_action="Report specific concerns to the Transport Office with details about routes and times.",
            who_to_contact="Transport Office",
            expected_timeline="3-5 days",
        )
    else:
        return StudentRecommendation(
            summary="We understand you have a transport concern.",
            immediate_action="Report the issue to the Transport Office with specific details: route, time, and nature of the problem. For urgent matters (breakdown, accident), contact them immediately.",
            who_to_contact="Transport Office",
            expected_timeline="3-7 days",
            additional_tips=[
                "Note the bus number, route, and time of incident",
                "For fare disputes, keep your ticket as evidence",
                "Report safety concerns (reckless driving) immediately",
            ],
        )


def _student_rec_catering(text, sentiment_type, urgency, categories, confidence):
    if sentiment_type == "positive":
        return StudentRecommendation(
            summary="Thank you for your positive feedback about catering services.",
            immediate_action="Share your positive experience with the Catering Manager.",
            who_to_contact="Catering Manager",
            expected_timeline="N/A",
        )
    elif sentiment_type == "mixed":
        return StudentRecommendation(
            summary="We understand you have mixed experiences with catering services.",
            immediate_action="Provide specific feedback to the Catering Manager about what's working and what needs improvement.",
            who_to_contact="Catering Manager",
            expected_timeline="3-5 days",
        )
    else:
        return StudentRecommendation(
            summary="We understand you have a concern about food or catering services.",
            immediate_action="Report the issue to the Catering Manager. For food quality concerns, keep a sample if possible. For hygiene issues, report immediately. For pricing concerns, request a price list.",
            who_to_contact="Catering Manager / Food Services",
            expected_timeline="1-3 days",
            additional_tips=[
                "For food quality issues, note the date, time, and specific meal",
                "For hygiene concerns (foreign objects in food), keep the evidence",
                "For pricing concerns, compare with the official price list",
                "Report food poisoning incidents immediately to Health Services",
            ],
        )


def _student_rec_library(text, sentiment_type, urgency, categories, confidence):
    if sentiment_type == "positive":
        return StudentRecommendation(
            summary="Thank you for your positive feedback about library services.",
            immediate_action="Share your positive experience with Library staff.",
            who_to_contact="Library Services",
            expected_timeline="N/A",
        )
    elif sentiment_type == "mixed":
        return StudentRecommendation(
            summary="We understand you have mixed experiences with library services.",
            immediate_action="Speak with the librarian about specific concerns while acknowledging what works well.",
            who_to_contact="Library Services",
            expected_timeline="3-5 days",
        )
    else:
        return StudentRecommendation(
            summary="We understand you have a concern about library services.",
            immediate_action="Speak with the librarian on duty about your concern. For book availability, check the catalog or request a reservation. For facility issues, report to Library Administration.",
            who_to_contact="Librarian / Library Administration",
            expected_timeline="3-7 days",
            additional_tips=[
                "Use the online catalog to check book availability before visiting",
                "For study space concerns, ask about alternative study areas",
                "For e-library issues, contact ICT if it's a technical problem",
                "Request books that are not available through the librarian",
            ],
        )


def _student_rec_student_affairs(text, sentiment_type, urgency, categories, confidence):
    if sentiment_type == "positive":
        return StudentRecommendation(
            summary="Thank you for your positive feedback about student affairs.",
            immediate_action="Get involved in more activities and encourage your peers.",
            who_to_contact="Student Affairs Office",
            expected_timeline="N/A",
        )
    elif sentiment_type == "mixed":
        return StudentRecommendation(
            summary="We understand you have mixed experiences with student affairs.",
            immediate_action="Contact the Student Affairs Office to discuss specific concerns and suggestions.",
            who_to_contact="Student Affairs Office",
            expected_timeline="3-5 days",
        )
    else:
        return StudentRecommendation(
            summary="We understand you have a concern about student affairs.",
            immediate_action="Contact the Student Affairs Office. For health concerns, visit the Health Center. For counseling, contact the Counseling Centre. For club/sports issues, speak with the Student Affairs Coordinator.",
            who_to_contact="Student Affairs Office / Health Center / Counseling Centre",
            expected_timeline="3-7 days",
            additional_tips=[
                "For health issues, visit the Health Center during opening hours",
                "For mental health support, the Counseling Centre offers confidential services",
                "For club registration, contact the Student Affairs Office",
                "For sports facilities, speak with the Sports Coordinator",
            ],
        )


def _student_rec_generic(text, sentiment_type, urgency, categories, confidence):
    if sentiment_type == "positive":
        return StudentRecommendation(
            summary="Thank you for your positive feedback.",
            immediate_action="We appreciate your kind words and will continue to improve.",
            who_to_contact="SRC Secretariat",
            expected_timeline="N/A",
        )
    elif sentiment_type == "mixed":
        return StudentRecommendation(
            summary="We understand you have mixed feedback.",
            immediate_action="Please provide more details so we can address the specific concerns.",
            who_to_contact="SRC Secretariat",
            expected_timeline="3-5 days",
        )
    else:
        return StudentRecommendation(
            summary="We have received your feedback and will review it.",
            immediate_action="If your concern is urgent, contact the SRC Secretariat directly. Otherwise, a representative will follow up with you.",
            who_to_contact="SRC Secretariat",
            expected_timeline="3-7 days",
        )


# ==================== ADMIN ACTION PLAN GENERATION ====================

def generate_admin_action_plan(
    text: str,
    category: str,
    sentiment_type: str,
    urgency: str,
    categories: List[CategoryMatch],
    confidence: float,
) -> AdminActionPlan:
    """Generate a comprehensive action plan for SRC/admin."""
    
    # Get category-specific action plan generators
    generators = {
        "Academics": _admin_plan_academics,
        "ICT": _admin_plan_ict,
        "Finance": _admin_plan_finance,
        "Safety": _admin_plan_safety,
        "Maintenance": _admin_plan_maintenance,
        "Accommodation": _admin_plan_accommodation,
        "Staff": _admin_plan_staff,
        "Administration": _admin_plan_administration,
        "Transport": _admin_plan_transport,
        "Catering": _admin_plan_catering,
        "Library": _admin_plan_library,
        "Student Affairs": _admin_plan_student_affairs,
    }
    
    generator = generators.get(category, _admin_plan_generic)
    return generator(text, sentiment_type, urgency, categories, confidence)


def _admin_plan_academics(text, sentiment_type, urgency, categories, confidence):
    dept = CATEGORY_DEFINITIONS["Academics"]["department"]
    if sentiment_type == "positive":
        return AdminActionPlan(
            investigation_steps=["Verify the positive feedback with the department", "Identify specific practices worth recognizing"],
            corrective_actions=["Acknowledge the good practice publicly", "Consider featuring as a model in faculty development"],
            preventive_actions=["Document best practices for sharing", "Include in faculty development workshops"],
            responsible_department=dept,
            priority_level="low",
            estimated_resolution_time="1-2 weeks",
            escalation_path="Dean of Faculty",
            monitoring_indicators=["Faculty evaluation scores", "Student satisfaction surveys"],
        )
    elif sentiment_type == "mixed":
        return AdminActionPlan(
            investigation_steps=["Review the specific concerns raised", "Compare with positive aspects mentioned", "Gather additional student feedback on the same issue"],
            corrective_actions=["Address the specific negative concerns with the department", "Reinforce the positive practices identified"],
            preventive_actions=["Establish regular feedback mechanisms", "Create a balanced recognition and improvement system"],
            responsible_department=dept,
            priority_level="medium",
            estimated_resolution_time="1-2 weeks",
            escalation_path="Dean of Faculty → Academic Board",
            monitoring_indicators=["Follow-up student feedback", "Faculty response to concerns"],
        )
    else:
        return AdminActionPlan(
            investigation_steps=["Review the specific academic concern with the department", "Check if other students have reported similar issues", "Examine relevant records (grades, attendance, etc.)"],
            corrective_actions=["Address the specific issue with the lecturer/department", "Ensure fair processes are followed", "Provide student with clear next steps"],
            preventive_actions=["Review teaching evaluation processes", "Establish clearer communication channels", "Implement regular academic quality checks"],
            responsible_department=dept,
            priority_level=urgency if urgency in ("critical", "high") else "medium",
            estimated_resolution_time="3-10 days",
            escalation_path="Department Head → Dean of Faculty → Academic Board",
            monitoring_indicators=["Number of similar complaints", "Resolution time", "Student satisfaction follow-up"],
        )


def _admin_plan_ict(text, sentiment_type, urgency, categories, confidence):
    dept = CATEGORY_DEFINITIONS["ICT"]["department"]
    if sentiment_type == "positive":
        return AdminActionPlan(
            investigation_steps=["Verify the positive feedback", "Identify what worked well"],
            corrective_actions=["Acknowledge the ICT team's effort", "Document the successful resolution"],
            preventive_actions=["Maintain current service levels", "Share best practices within the team"],
            responsible_department=dept,
            priority_level="low",
            estimated_resolution_time="1 week",
            escalation_path="ICT Director",
            monitoring_indicators=["Network uptime", "User satisfaction scores"],
        )
    elif sentiment_type == "mixed":
        return AdminActionPlan(
            investigation_steps=["Identify the specific ICT issues raised", "Check system logs for reported problems", "Compare with positive aspects mentioned"],
            corrective_actions=["Address the specific technical issues", "Maintain the aspects that are working well"],
            preventive_actions=["Implement monitoring for the identified issues", "Regular system health checks"],
            responsible_department=dept,
            priority_level="medium",
            estimated_resolution_time="2-5 days",
            escalation_path="ICT Director → Dean of IT",
            monitoring_indicators=["System uptime", "Issue resolution time", "Repeat complaints"],
        )
    else:
        return AdminActionPlan(
            investigation_steps=["Check system logs for the reported issue", "Verify the scope (individual vs. widespread)", "Identify root cause (hardware, software, network)"],
            corrective_actions=["Resolve the immediate technical issue", "Provide workaround if fix takes time", "Communicate status to affected users"],
            preventive_actions=["Implement monitoring alerts", "Schedule preventive maintenance", "Update documentation and user guides"],
            responsible_department=dept,
            priority_level=urgency if urgency in ("critical", "high") else "medium",
            estimated_resolution_time="1-5 days",
            escalation_path="ICT Director → Dean of IT → Vice Chancellor",
            monitoring_indicators=["System uptime", "Number of similar issues", "Mean time to resolution"],
        )


def _admin_plan_finance(text, sentiment_type, urgency, categories, confidence):
    dept = CATEGORY_DEFINITIONS["Finance"]["department"]
    if sentiment_type == "positive":
        return AdminActionPlan(
            investigation_steps=["Verify the positive feedback", "Identify what financial process worked well"],
            corrective_actions=["Acknowledge the Finance Office effort", "Document the successful process"],
            preventive_actions=["Maintain current service standards", "Share best practices"],
            responsible_department=dept,
            priority_level="low",
            estimated_resolution_time="1 week",
            escalation_path="Finance Director",
            monitoring_indicators=["Payment processing time", "Student satisfaction"],
        )
    elif sentiment_type == "mixed":
        return AdminActionPlan(
            investigation_steps=["Review the specific financial concerns", "Compare with positive aspects", "Check fee structures and processes"],
            corrective_actions=["Address specific billing/payment issues", "Maintain efficient processes"],
            preventive_actions=["Review fee communication", "Improve payment options"],
            responsible_department=dept,
            priority_level="medium",
            estimated_resolution_time="1-2 weeks",
            escalation_path="Finance Director → Bursar",
            monitoring_indicators=["Payment dispute resolution time", "Student feedback"],
        )
    else:
        return AdminActionPlan(
            investigation_steps=["Review the student's financial records", "Verify fee calculations and payment history", "Check for any system errors or miscommunications"],
            corrective_actions=["Correct any billing errors immediately", "Process pending refunds or adjustments", "Provide clear explanation of charges"],
            preventive_actions=["Improve fee communication to students", "Implement better payment tracking", "Review financial aid disbursement processes"],
            responsible_department=dept,
            priority_level=urgency if urgency in ("critical", "high") else "medium",
            estimated_resolution_time="3-10 days",
            escalation_path="Finance Director → Bursar → Vice Chancellor",
            monitoring_indicators=["Number of financial disputes", "Resolution time", "Student satisfaction"],
        )


def _admin_plan_safety(text, sentiment_type, urgency, categories, confidence):
    dept = CATEGORY_DEFINITIONS["Safety"]["department"]
    if sentiment_type == "positive":
        return AdminActionPlan(
            investigation_steps=["Verify the positive feedback", "Identify what security measure worked well"],
            corrective_actions=["Acknowledge the security team's effort", "Document the successful intervention"],
            preventive_actions=["Maintain current security standards", "Share best practices"],
            responsible_department=dept,
            priority_level="low",
            estimated_resolution_time="1 week",
            escalation_path="Security Head",
            monitoring_indicators=["Incident reports", "Response time"],
        )
    elif sentiment_type == "mixed":
        return AdminActionPlan(
            investigation_steps=["Review the specific safety concerns", "Compare with positive security aspects", "Assess current security measures"],
            corrective_actions=["Address specific security gaps", "Maintain effective measures"],
            preventive_actions=["Review patrol schedules", "Assess surveillance coverage"],
            responsible_department=dept,
            priority_level="high",
            estimated_resolution_time="3-7 days",
            escalation_path="Security Head → Registrar → Vice Chancellor",
            monitoring_indicators=["Security incident trends", "Student safety perception"],
        )
    else:
        priority = "critical" if urgency == "critical" else "high"
        return AdminActionPlan(
            investigation_steps=["Verify the incident report immediately", "Gather evidence (CCTV, witness statements)", "Assess the scope and ongoing risk", "Coordinate with local law enforcement if needed"],
            corrective_actions=["Deploy additional security if needed", "Increase patrols in affected area", "Issue safety advisory to students", "Provide support to affected individuals"],
            preventive_actions=["Review and upgrade security measures", "Improve lighting in vulnerable areas", "Conduct safety awareness campaigns", "Review access control procedures"],
            responsible_department=dept,
            priority_level=priority,
            estimated_resolution_time="Immediate - 7 days",
            escalation_path="Security Head → Registrar → Vice Chancellor → Law Enforcement",
            monitoring_indicators=["Number of security incidents", "Response time", "Student safety perception survey"],
        )


def _admin_plan_maintenance(text, sentiment_type, urgency, categories, confidence):
    dept = CATEGORY_DEFINITIONS["Maintenance"]["department"]
    if sentiment_type == "positive":
        return AdminActionPlan(
            investigation_steps=["Verify the positive feedback", "Identify what maintenance work was done well"],
            corrective_actions=["Acknowledge the maintenance team's effort", "Document the successful repair"],
            preventive_actions=["Maintain current service standards", "Schedule preventive maintenance"],
            responsible_department=dept,
            priority_level="low",
            estimated_resolution_time="1 week",
            escalation_path="Maintenance Head",
            monitoring_indicators=["Repair completion time", "Repeat issues"],
        )
    elif sentiment_type == "mixed":
        return AdminActionPlan(
            investigation_steps=["Review the specific maintenance concerns", "Compare with positive aspects", "Assess current maintenance backlog"],
            corrective_actions=["Address specific maintenance issues", "Maintain completed work"],
            preventive_actions=["Review maintenance response times", "Improve preventive maintenance schedule"],
            responsible_department=dept,
            priority_level="medium",
            estimated_resolution_time="1-2 weeks",
            escalation_path="Maintenance Head → Facilities Director",
            monitoring_indicators=["Maintenance request volume", "Resolution time"],
        )
    else:
        return AdminActionPlan(
            investigation_steps=["Inspect the reported issue on-site", "Assess the scope and urgency", "Identify required resources and materials"],
            corrective_actions=["Dispatch maintenance team for immediate repair", "Provide temporary solution if full repair takes time", "Communicate timeline to affected users"],
            preventive_actions=["Implement preventive maintenance schedule", "Conduct regular facility audits", "Establish quick-response protocols for common issues"],
            responsible_department=dept,
            priority_level=urgency if urgency in ("critical", "high") else "medium",
            estimated_resolution_time="2-7 days",
            escalation_path="Maintenance Head → Facilities Director → Registrar",
            monitoring_indicators=["Maintenance request volume", "Resolution time", "Repeat issues"],
        )


def _admin_plan_accommodation(text, sentiment_type, urgency, categories, confidence):
    dept = CATEGORY_DEFINITIONS["Accommodation"]["department"]
    if sentiment_type == "positive":
        return AdminActionPlan(
            investigation_steps=["Verify the positive feedback", "Identify what accommodation aspect worked well"],
            corrective_actions=["Acknowledge Hall Management's effort", "Document the successful approach"],
            preventive_actions=["Maintain current standards", "Share best practices across halls"],
            responsible_department=dept,
            priority_level="low",
            estimated_resolution_time="1 week",
            escalation_path="Hall Management",
            monitoring_indicators=["Occupancy rates", "Student satisfaction"],
        )
    elif sentiment_type == "mixed":
        return AdminActionPlan(
            investigation_steps=["Review specific accommodation concerns", "Compare with positive aspects", "Assess current hall conditions"],
            corrective_actions=["Address specific issues (maintenance, noise, etc.)", "Maintain positive aspects"],
            preventive_actions=["Review hall management policies", "Improve communication with residents"],
            responsible_department=dept,
            priority_level="medium",
            estimated_resolution_time="1-2 weeks",
            escalation_path="Hall Management → Accommodation Officer",
            monitoring_indicators=["Complaint trends", "Student satisfaction"],
        )
    else:
        return AdminActionPlan(
            investigation_steps=["Inspect the reported accommodation issue", "Verify the scope (individual room vs. widespread)", "Check maintenance records for the location"],
            corrective_actions=["Address the immediate issue (repair, cleaning, pest control)", "Mediate conflicts if needed", "Provide temporary relocation if necessary"],
            preventive_actions=["Improve room allocation process", "Implement regular hall inspections", "Establish clear visitation and noise policies"],
            responsible_department=dept,
            priority_level=urgency if urgency in ("critical", "high") else "medium",
            estimated_resolution_time="2-7 days",
            escalation_path="Hall Management → Accommodation Officer → Dean of Students",
            monitoring_indicators=["Number of accommodation complaints", "Resolution time", "Student satisfaction"],
        )


def _admin_plan_staff(text, sentiment_type, urgency, categories, confidence):
    dept = CATEGORY_DEFINITIONS["Staff"]["department"]
    if sentiment_type == "positive":
        return AdminActionPlan(
            investigation_steps=["Verify the positive feedback", "Identify the staff member and specific good conduct"],
            corrective_actions=["Acknowledge the staff member's good conduct", "Consider for recognition/award"],
            preventive_actions=["Share best practices", "Include in staff development"],
            responsible_department=dept,
            priority_level="low",
            estimated_resolution_time="1-2 weeks",
            escalation_path="Department Head → HR",
            monitoring_indicators=["Staff evaluation scores", "Student feedback"],
        )
    elif sentiment_type == "mixed":
        return AdminActionPlan(
            investigation_steps=["Review the specific conduct concerns", "Compare with positive aspects", "Gather additional feedback"],
            corrective_actions=["Address specific conduct issues", "Reinforce positive behaviors"],
            preventive_actions=["Review staff conduct policies", "Implement regular feedback mechanisms"],
            responsible_department=dept,
            priority_level="medium",
            estimated_resolution_time="1-2 weeks",
            escalation_path="Department Head → HR → Registrar",
            monitoring_indicators=["Staff conduct complaints", "Resolution outcomes"],
        )
    else:
        return AdminActionPlan(
            investigation_steps=["Investigate the specific incident(s) reported", "Gather evidence and witness statements", "Review the staff member's conduct history", "Provide due process for the staff member"],
            corrective_actions=["Take appropriate disciplinary action if verified", "Provide support to the affected student", "Implement immediate corrective measures"],
            preventive_actions=["Review staff conduct policies", "Implement regular training on professional conduct", "Establish clearer complaint procedures"],
            responsible_department=dept,
            priority_level=urgency if urgency in ("critical", "high") else "medium",
            estimated_resolution_time="5-14 days",
            escalation_path="Department Head → HR → Registrar → Vice Chancellor",
            monitoring_indicators=["Number of conduct complaints", "Resolution time", "Repeat offenses"],
        )


def _admin_plan_administration(text, sentiment_type, urgency, categories, confidence):
    dept = CATEGORY_DEFINITIONS["Administration"]["department"]
    if sentiment_type == "positive":
        return AdminActionPlan(
            investigation_steps=["Verify the positive feedback", "Identify what administrative process worked well"],
            corrective_actions=["Acknowledge the administrative staff's effort", "Document the successful process"],
            preventive_actions=["Maintain current service standards", "Share best practices"],
            responsible_department=dept,
            priority_level="low",
            estimated_resolution_time="1 week",
            escalation_path="Administration Head",
            monitoring_indicators=["Processing time", "Student satisfaction"],
        )
    elif sentiment_type == "mixed":
        return AdminActionPlan(
            investigation_steps=["Review specific administrative concerns", "Compare with positive aspects", "Assess current processes"],
            corrective_actions=["Address specific process issues", "Maintain efficient processes"],
            preventive_actions=["Review administrative procedures", "Improve communication with students"],
            responsible_department=dept,
            priority_level="medium",
            estimated_resolution_time="1-2 weeks",
            escalation_path="Administration Head → Registrar",
            monitoring_indicators=["Processing times", "Complaint trends"],
        )
    else:
        return AdminActionPlan(
            investigation_steps=["Review the specific administrative issue", "Check processing records and timelines", "Identify bottlenecks or errors"],
            corrective_actions=["Resolve the immediate issue (process document, correct error)", "Provide clear timeline for resolution", "Communicate status to the student"],
            preventive_actions=["Review and streamline administrative processes", "Implement tracking systems for student requests", "Establish service level agreements"],
            responsible_department=dept,
            priority_level=urgency if urgency in ("critical", "high") else "medium",
            estimated_resolution_time="3-14 days",
            escalation_path="Administration Head → Registrar → Vice Chancellor",
            monitoring_indicators=["Processing times", "Number of pending requests", "Student satisfaction"],
        )


def _admin_plan_transport(text, sentiment_type, urgency, categories, confidence):
    dept = CATEGORY_DEFINITIONS["Transport"]["department"]
    if sentiment_type == "positive":
        return AdminActionPlan(
            investigation_steps=["Verify the positive feedback", "Identify what transport service worked well"],
            corrective_actions=["Acknowledge the transport team's effort", "Document the successful service"],
            preventive_actions=["Maintain current service standards", "Share best practices"],
            responsible_department=dept,
            priority_level="low",
            estimated_resolution_time="1 week",
            escalation_path="Transport Head",
            monitoring_indicators=["Service reliability", "Student satisfaction"],
        )
    elif sentiment_type == "mixed":
        return AdminActionPlan(
            investigation_steps=["Review specific transport concerns", "Compare with positive aspects", "Assess current routes and schedules"],
            corrective_actions=["Address specific service issues", "Maintain reliable services"],
            preventive_actions=["Review route efficiency", "Improve schedule communication"],
            responsible_department=dept,
            priority_level="medium",
            estimated_resolution_time="1-2 weeks",
            escalation_path="Transport Head → Registrar",
            monitoring_indicators=["Service reliability metrics", "Complaint trends"],
        )
    else:
        return AdminActionPlan(
            investigation_steps=["Verify the reported transport issue", "Check vehicle/driver records", "Assess the scope (single incident vs. systemic)"],
            corrective_actions=["Address the immediate issue (repair vehicle, reassign driver)", "Provide alternative transport if needed", "Communicate status to affected students"],
            preventive_actions=["Review transport schedules and routes", "Implement regular vehicle maintenance", "Establish driver conduct policies"],
            responsible_department=dept,
            priority_level=urgency if urgency in ("critical", "high") else "medium",
            estimated_resolution_time="3-10 days",
            escalation_path="Transport Head → Registrar → Vice Chancellor",
            monitoring_indicators=["Service reliability", "Number of complaints", "Vehicle condition"],
        )


def _admin_plan_catering(text, sentiment_type, urgency, categories, confidence):
    dept = CATEGORY_DEFINITIONS["Catering"]["department"]
    if sentiment_type == "positive":
        return AdminActionPlan(
            investigation_steps=["Verify the positive feedback", "Identify what catering aspect worked well"],
            corrective_actions=["Acknowledge the catering team's effort", "Document the successful practice"],
            preventive_actions=["Maintain current food quality standards", "Share best practices"],
            responsible_department=dept,
            priority_level="low",
            estimated_resolution_time="1 week",
            escalation_path="Catering Manager",
            monitoring_indicators=["Food quality scores", "Student satisfaction"],
        )
    elif sentiment_type == "mixed":
        return AdminActionPlan(
            investigation_steps=["Review specific catering concerns", "Compare with positive aspects", "Assess current food quality and service"],
            corrective_actions=["Address specific food quality issues", "Maintain good practices"],
            preventive_actions=["Review food preparation processes", "Improve menu variety"],
            responsible_department=dept,
            priority_level="medium",
            estimated_resolution_time="1-2 weeks",
            escalation_path="Catering Manager → Health & Safety",
            monitoring_indicators=["Food quality scores", "Complaint trends"],
        )
    else:
        return AdminActionPlan(
            investigation_steps=["Inspect the reported food issue immediately", "Check kitchen hygiene and food preparation", "Review food handling procedures", "Check for any health incidents"],
            corrective_actions=["Address the immediate food safety issue", "Dispose of any unsafe food", "Increase hygiene measures"],
            preventive_actions=["Implement regular health inspections", "Review food supplier standards", "Establish food quality monitoring", "Conduct staff hygiene training"],
            responsible_department=dept,
            priority_level=urgency if urgency in ("critical", "high") else "medium",
            estimated_resolution_time="1-5 days",
            escalation_path="Catering Manager → Health & Safety → Registrar",
            monitoring_indicators=["Food safety incidents", "Health inspection scores", "Student complaints"],
        )


def _admin_plan_library(text, sentiment_type, urgency, categories, confidence):
    dept = CATEGORY_DEFINITIONS["Library"]["department"]
    if sentiment_type == "positive":
        return AdminActionPlan(
            investigation_steps=["Verify the positive feedback", "Identify what library service worked well"],
            corrective_actions=["Acknowledge the library staff's effort", "Document the successful service"],
            preventive_actions=["Maintain current service standards", "Share best practices"],
            responsible_department=dept,
            priority_level="low",
            estimated_resolution_time="1 week",
            escalation_path="Head Librarian",
            monitoring_indicators=["Library usage statistics", "Student satisfaction"],
        )
    elif sentiment_type == "mixed":
        return AdminActionPlan(
            investigation_steps=["Review specific library concerns", "Compare with positive aspects", "Assess current library resources and services"],
            corrective_actions=["Address specific resource/service issues", "Maintain good services"],
            preventive_actions=["Review resource allocation", "Improve study facilities"],
            responsible_department=dept,
            priority_level="medium",
            estimated_resolution_time="1-2 weeks",
            escalation_path="Head Librarian → Academic Affairs",
            monitoring_indicators=["Resource availability", "Student satisfaction"],
        )
    else:
        return AdminActionPlan(
            investigation_steps=["Verify the reported library issue", "Check resource availability and condition", "Assess facility conditions", "Review library policies"],
            corrective_actions=["Address the immediate issue (acquire resources, fix facilities)", "Provide alternative solutions", "Update policies if needed"],
            preventive_actions=["Review and update library resources", "Improve study facilities", "Extend opening hours if needed", "Enhance e-library services"],
            responsible_department=dept,
            priority_level=urgency if urgency in ("critical", "high") else "medium",
            estimated_resolution_time="3-14 days",
            escalation_path="Head Librarian → Academic Affairs → Registrar",
            monitoring_indicators=["Resource availability", "Facility condition", "Student satisfaction"],
        )


def _admin_plan_student_affairs(text, sentiment_type, urgency, categories, confidence):
    dept = CATEGORY_DEFINITIONS["Student Affairs"]["department"]
    if sentiment_type == "positive":
        return AdminActionPlan(
            investigation_steps=["Verify the positive feedback", "Identify what student affair aspect worked well"],
            corrective_actions=["Acknowledge the effort", "Document the successful approach"],
            preventive_actions=["Maintain current standards", "Share best practices"],
            responsible_department=dept,
            priority_level="low",
            estimated_resolution_time="1 week",
            escalation_path="Dean of Students",
            monitoring_indicators=["Student participation", "Satisfaction scores"],
        )
    elif sentiment_type == "mixed":
        return AdminActionPlan(
            investigation_steps=["Review specific concerns", "Compare with positive aspects", "Assess current programs and services"],
            corrective_actions=["Address specific issues", "Maintain good programs"],
            preventive_actions=["Review program effectiveness", "Improve service delivery"],
            responsible_department=dept,
            priority_level="medium",
            estimated_resolution_time="1-2 weeks",
            escalation_path="Dean of Students → Registrar",
            monitoring_indicators=["Program participation", "Student feedback"],
        )
    else:
        return AdminActionPlan(
            investigation_steps=["Verify the reported issue", "Assess the scope and impact", "Review current programs and resources"],
            corrective_actions=["Address the immediate issue", "Provide support to affected students", "Allocate additional resources if needed"],
            preventive_actions=["Review and improve programs", "Enhance health and counseling services", "Improve sports and recreation facilities"],
            responsible_department=dept,
            priority_level=urgency if urgency in ("critical", "high") else "medium",
            estimated_resolution_time="3-14 days",
            escalation_path="Dean of Students → Registrar → Vice Chancellor",
            monitoring_indicators=["Student participation", "Service utilization", "Satisfaction scores"],
        )


def _admin_plan_generic(text, sentiment_type, urgency, categories, confidence):
    if sentiment_type == "positive":
        return AdminActionPlan(
            investigation_steps=["Verify the positive feedback"],
            corrective_actions=["Acknowledge and document the positive feedback"],
            preventive_actions=["Maintain current standards"],
            responsible_department="SRC Secretariat",
            priority_level="low",
            estimated_resolution_time="1 week",
            escalation_path="SRC President",
            monitoring_indicators=["Student satisfaction"],
        )
    elif sentiment_type == "mixed":
        return AdminActionPlan(
            investigation_steps=["Review the feedback for specific concerns", "Identify positive aspects to maintain"],
            corrective_actions=["Address specific concerns", "Reinforce positive aspects"],
            preventive_actions=["Establish regular feedback mechanisms"],
            responsible_department="SRC Secretariat",
            priority_level="medium",
            estimated_resolution_time="1-2 weeks",
            escalation_path="SRC President",
            monitoring_indicators=["Follow-up feedback"],
        )
    else:
        return AdminActionPlan(
            investigation_steps=["Review the feedback in detail", "Gather additional information if needed", "Identify the responsible department"],
            corrective_actions=["Address the immediate concern", "Coordinate with relevant department"],
            preventive_actions=["Review policies and procedures", "Implement improvements"],
            responsible_department="SRC Secretariat",
            priority_level=urgency if urgency in ("critical", "high") else "medium",
            estimated_resolution_time="3-10 days",
            escalation_path="SRC President → Registrar",
            monitoring_indicators=["Resolution time", "Student satisfaction"],
        )


# ==================== MAIN RECOMMENDATION FUNCTION ====================

def generate_recommendation(
    text: str,
    category: Optional[str] = None,
    urgency_score: Optional[int] = None,
    sentiment: Optional[str] = None,
    sentiment_score: Optional[float] = None,
    emotion: Optional[dict] = None,
    db_templates: Optional[List[dict]] = None,
) -> RecommendationResult:
    """
    Generate a complete recommendation with separate student and admin layers.
    
    This is the main entry point for the recommendation engine.
    """
    # If sentiment is not provided, analyze the text directly
    if sentiment is None or sentiment_score is None:
        try:
            from sentiment_analyzer import process_feedback
            analysis = process_feedback(text)
            sentiment = analysis['sentiment']
            sentiment_score = analysis['sentiment_score']
            if urgency_score is None:
                urgency_score = analysis['urgency_score']
        except Exception:
            # If sentiment analysis fails, use defaults
            sentiment = sentiment or "Neutral"
            sentiment_score = sentiment_score or 0.0
    
    # Step 1: Classify categories
    categories = classify_categories(text)
    primary_category = categories[0].name if categories else "Other"
    
    # If a category was provided by the user, use it as primary
    if category and category != "Other" and category in CATEGORY_DEFINITIONS:
        # Reorder categories to put the user-selected one first
        user_cat = None
        for cat in categories:
            if cat.name == category:
                user_cat = cat
                break
        if user_cat:
            categories.remove(user_cat)
            categories.insert(0, user_cat)
            user_cat.is_primary = True
            primary_category = category
    
    # Step 2: Determine sentiment type
    sentiment_label = sentiment or "Neutral"
    sentiment_val = sentiment_score or 0.0
    sentiment_type = analyze_sentiment_type(text, sentiment_label, sentiment_val)
    
    # Step 3: Determine urgency
    urgency = determine_urgency(urgency_score or 3, sentiment_type, categories, text)
    
    # Step 4: Calculate confidence
    confidence = categories[0].confidence if categories else 0.0
    # Adjust confidence based on sentiment clarity
    if sentiment_type == "unclear":
        confidence *= 0.5
    elif sentiment_type == "mixed":
        confidence *= 0.8
    
    # Step 5: Check for multi-issue feedback
    # Use a lower threshold to detect multiple significant categories
    significant_categories = [c for c in categories if c.confidence > 0.15]
    multi_issue = len(significant_categories) > 1
    
    # Step 6: Generate student recommendation
    student_rec = generate_student_recommendation(
        text, primary_category, sentiment_type, urgency, categories, confidence
    )
    
    # Step 7: Generate admin action plan
    admin_plan = generate_admin_action_plan(
        text, primary_category, sentiment_type, urgency, categories, confidence
    )
    
    # Step 8: Check if fallback is needed
    fallback_used = False
    fallback_message = None
    if confidence < FALLBACK_CONFIDENCE_THRESHOLD:
        fallback_used = True
        fallback_message = (
            "The system could not confidently categorize this feedback. "
            "Please provide more specific details about your concern, including: "
            "(1) What the issue is, (2) Where it occurred, (3) When it happened, "
            "(4) Who is involved. An SRC representative will review your feedback manually."
        )
        # Override with fallback recommendations
        student_rec = StudentRecommendation(
            summary="We received your feedback but need more specific information.",
            immediate_action="Please provide more details: what the issue is, where it occurred, when it happened, and who is involved.",
            who_to_contact="SRC Secretariat",
            expected_timeline="3-5 business days after more details are provided",
        )
        admin_plan = AdminActionPlan(
            investigation_steps=["Review the feedback manually", "Contact the student for clarification"],
            corrective_actions=["Await clarification before taking action"],
            preventive_actions=["Improve feedback collection to capture more details"],
            responsible_department="SRC Secretariat",
            priority_level="medium",
            estimated_resolution_time="5-10 days",
            escalation_path="SRC President",
            monitoring_indicators=["Clarification received", "Issue categorized"],
        )
    
    return RecommendationResult(
        primary_category=primary_category,
        all_categories=categories,
        sentiment=sentiment_type,
        urgency=urgency,
        student_recommendation=student_rec,
        admin_action_plan=admin_plan,
        confidence=round(confidence, 3),
        fallback_used=fallback_used,
        fallback_message=fallback_message,
        multi_issue=multi_issue,
    )
