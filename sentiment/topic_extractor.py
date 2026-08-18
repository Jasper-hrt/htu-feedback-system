"""Lightweight campus-topic extraction for dashboard summaries.

This is deliberately deterministic and dependency-free. It extracts meaningful
campus topics from feedback without replacing the sentiment engine.
"""
from __future__ import annotations
import re

TOPIC_PATTERNS = [
    ("Wi-Fi / Internet", r"\b(wi[\s-]?fi|wifi|internet|network|hotspot|data)\b"),
    ("Power / Electricity", r"\b(power|electricity|light|lights|blackout|dumsor)\b"),
    ("Water / Plumbing", r"\b(water|tap|pipe|plumbing|leak|leaking)\b"),
    ("Flooding / Drainage", r"\b(flood|flooding|flooded|drainage|drain|waterlogging)\b"),
    ("Hostel", r"\b(hostel|residence|hall|dorm)\b"),
    ("Security / Safety", r"\b(robbery|robbed|steal|stolen|violence|assault|gunshot|shooting|attack|threat|kidnap)\b"),
    ("Lecturers / Teaching", r"\b(lecturer|teacher|course|class|lecture|teaching|assignment)\b"),
    ("Exams / Registration", r"\b(exam|examination|registration|results|result|course registration)\b"),
    ("Fees / Finance", r"\b(fee|fees|payment|school fees|finance|refund)\b"),
    ("Facilities / Maintenance", r"\b(classroom|building|toilet|washroom|furniture|chair|desk|maintenance|broken|repair|facility)\b"),
    ("Library", r"\b(library|book|books|study area)\b"),
    ("Transport", r"\b(bus|transport|shuttle|taxi|parking)\b"),
]

def extract_topics(text: str, limit: int = 3) -> list[str]:
    text = str(text or "").lower()
    found = []
    for label, pattern in TOPIC_PATTERNS:
        if re.search(pattern, text):
            found.append(label)
            if len(found) >= limit:
                break
    return found
