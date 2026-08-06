"""
hybrid_engine.py

Main Hybrid Sentiment Analysis Engine.

Combines:
- Text preprocessing (Phase 2)
- VADER, TextBlob, AFINN, SentiWordNet, Custom Lexicon
- Emotion Detection with intensity weighting (Phase 1)
- Decision Engine with confidence-weighted voting (Phase 3)
- Confidence Calculator with agreement matrix (Phase 3)
- Unknown Word Detection
- Aspect-based Sentiment Extraction (Phase 4)
- Sarcasm Detection (Phase 4)
- Intensity Modifier Handling (Phase 4)
"""

import re
from typing import Dict, List

from sentiment.vader_engine import VaderEngine
from sentiment.textblob_engine import TextBlobEngine
from sentiment.afinn_engine import AfinnEngine
from sentiment.sentiwordnet_engine import SentiWordNetEngine

from sentiment.emotion_engine import NRCEmotionAnalyzer
from sentiment.decision_engine import DecisionEngine
from sentiment.confidence import ConfidenceCalculator

from sentiment.preprocessing import TextPreprocessor
from sentiment.custom_lexicon import CustomLexiconManager
from sentiment.unknown_detector import UnknownWordDetector


# ================================================================
# ASPECT CATEGORIES
# ================================================================
ASPECT_KEYWORDS = {
    "accommodation": [
        "hostel", "dorm", "room", "hall", "bed", "bathroom", "shower",
        "toilet", "water", "flood", "leak", "plumbing", "sewage",
        "drainage", "cockroach", "rodent", "infestation", "mold",
        "ventilation", "stuffy", "overcrowding", "accommodation",
        "mattress", "cupboard", "window", "door", "lock", "key",
        "electricity", "light", "socket", "fan", "switch"
    ],
    "ict_wifi": [
        "wifi", "wi-fi", "internet", "network", "connection", "computer",
        "lab", "portal", "server", "slow", "disconnect", "offline",
        "lagging", "buffering", "bandwidth", "data", "signal", "hotspot",
        "speed", "download", "upload", "browsing", "connectivity",
        "email", "password", "ict", "software", "hardware"
    ],
    "academics": [
        "lecturer", "professor", "class", "exam", "course", "assignment",
        "timetable", "curriculum", "syllabus", "grade", "result", "score",
        "assessment", "quiz", "test", "project", "lecture", "tutorial",
        "seminar", "workshop", "study", "teaching", "education",
        "department", "faculty", "dean", "coordinator", "semester"
    ],
    "catering": [
        "food", "canteen", "cafeteria", "meal", "dining", "kitchen",
        "restaurant", "cook", "menu", "breakfast", "lunch", "dinner",
        "snack", "drink", "rice", "banku", "fufu", "kenkey", "jollof",
        "waakye", "beans", "chicken", "fish", "meat", "soup", "stew",
        "price", "expensive", "tasty", "bland", "stale", "portion",
        "hungry", "starving", "hygiene", "plate", "bowl"
    ],
    "facilities": [
        "library", "classroom", "building", "elevator", "lift",
        "ac", "air conditioner", "chair", "desk", "projector", "board",
        "speaker", "microphone", "lighting", "fan", "generator",
        "sports", "gym", "football", "basketball", "court", "field",
        "auditorium", "lounge", "study area", "parking"
    ],
    "safety": [
        "security", "safe", "theft", "stolen", "robbery", "dark",
        "lighting", "patrol", "gate", "guard", "cctv", "camera",
        "danger", "unsafe", "gunshots", "shooting", "weapon", "violence",
        "assault", "harassment", "emergency", "threat", "attack",
        "fight", "blood", "injured", "injury", "panic", "hazard",
        "intruder", "suspicious", "vandalism", "fire"
    ],
    "transport": [
        "bus", "shuttle", "parking", "transport", "vehicle", "car",
        "driver", "fuel", "trotro", "taxi", "campus", "road",
        "pedestrian", "route", "schedule", "delay", "cancel"
    ],
    "mental_health": [
        "stress", "stressed", "anxiety", "anxious", "counseling",
        "counselling", "mental", "wellness", "wellbeing",
        "support", "depression", "depressed", "pressure",
        "overwhelmed", "burnout", "burned out", "exhausted",
        "tired", "homesick", "lonely", "isolated", "worried",
        "scared", "afraid", "therapy", "therapist", "health center",
        "clinic", "hospital", "doctor", "nurse", "medicine"
    ],
    "administration": [
        "admin", "administration", "registry", "registrar", "fee",
        "school fees", "payment", "invoice", "receipt", "admission",
        "enrollment", "transcript", "certificate", "document",
        "complaint", "response", "feedback", "resolution",
        "office", "staff", "secretary", "reception", "front desk",
        "help desk", "service", "support"
    ],
    "cleanliness": [
        "clean", "cleanliness", "hygiene", "sanitary", "sanitation",
        "dirty", "filthy", "messy", "trash", "rubbish", "garbage",
        "waste", "bin", "dump", "sweep", "mop", "janitor", "cleaner",
        "housekeeping", "maintenance", "repair", "broken", "damaged",
        "faulty", "neglected", "unkempt", "tidy", "organized"
    ]
}


# ================================================================
# SARACSM DETECTION PATTERNS
# ================================================================
SARACSM_PATTERNS = [
    # Positive word followed by negative context
    (r"(?:great|excellent|wonderful|fantastic|amazing|lovely|perfect|brilliant)\s+(?:job|work|service|system|experience).*(?:not|never|no|but|however|yet|still)", 0.7),
    # Sarcastic thanks
    (r"(?:thanks|thank you)\s+(?:for\s+)?(?:nothing|no\s+help|wasting|ruining)", 0.8),
    # "What a ..." patterns
    (r"what\s+a\s+(?:joke|shame|disaster|mess|waste|surprise)", 0.6),
    # "Yeah right" / "Sure" sarcasm
    (r"\byeah\s+right\b", 0.6),
    (r"\bsure\s+jan\b|\bsure\s+feb\b|\bsure\s+mar\b", 0.5),
    # Rhetorical frustration
    (r"(?:is\s+it\s+too\s+much\s+to\s+ask|how\s+hard\s+is\s+it|is\s+this\s+too\s+much)", 0.5),
    # "Nice" + negative word nearby
    (r"\bnice\b.{0,30}(?:not|never|no|broken|terrible|awful|useless)", 0.5),
    # "Just what I needed"
    (r"just\s+what\s+(?:we|i)\s+needed", 0.5),
    # "Another day, another problem"
    (r"another\s+day.*another\s+(?:problem|issue|delay|failure)", 0.4),
    # Sarcastic "good luck"
    (r"good\s+luck\s+with\s+that", 0.5),
    # "Oh really" 
    (r"\boh\s+really\b", 0.4),
    # Positive verb + nothing
    (r"(?:love|like|enjoy)\s+(?:that|how|when|the\s+way).*(?:not|never|no|doesn|don|didn|won)", 0.6),
]


# ================================================================
# INTENSITY MODIFIERS
# ================================================================
INTENSITY_MODIFIERS = {
    # Boosters
    "very": 1.5, "extremely": 2.0, "absolutely": 2.0,
    "completely": 1.8, "totally": 1.8, "highly": 1.6,
    "seriously": 1.5, "really": 1.3, "so": 1.3, "such": 1.3,
    "too": 1.2, "incredibly": 1.8, "unbelievably": 1.8,
    "exceptionally": 1.7, "extraordinarily": 1.8,
    "terribly": 1.5, "awfully": 1.4, "horribly": 1.5,
    "utterly": 1.8, "deeply": 1.5, "greatly": 1.3,
    "immensely": 1.6, "massively": 1.5, "hugely": 1.5,
    "heavily": 1.4, "strongly": 1.4, "badly": 1.3,
    "fiercely": 1.5, "wildly": 1.4,
    "very very": 2.0, "super": 1.3, "mad": 1.3, "hella": 1.5,
    # Diminishers
    "barely": 0.3, "hardly": 0.3, "scarcely": 0.3,
    "slightly": 0.5, "somewhat": 0.5, "a bit": 0.5,
    "a little": 0.5, "kind of": 0.5, "sort of": 0.5,
    "almost": 0.6, "nearly": 0.6, "marginally": 0.4,
    "mildly": 0.5, "moderately": 0.6, "fairly": 0.6,
    "pretty": 0.7, "quite": 0.7, "rather": 0.6,
    "relatively": 0.6, "reasonably": 0.6,
    "mostly": 0.7, "largely": 0.7, "partially": 0.5,
    "a little bit": 0.4,
}


class HybridSentimentEngine:

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.vader = VaderEngine()
        self.textblob = TextBlobEngine()
        self.afinn = AfinnEngine()
        self.sentiwordnet = SentiWordNetEngine()
        self.emotion = NRCEmotionAnalyzer()
        self.decision = DecisionEngine()
        self.confidence = ConfidenceCalculator()
        self.lexicon = CustomLexiconManager()
        self.unknown = UnknownWordDetector()

    def analyze(self, text: str) -> dict:
        """
        Analyze student feedback using multiple
        sentiment-analysis methods with context-aware features.
        """
        if not text or not text.strip():
            return self._empty_result(text)

        # STEP 1: Clean
        cleaned_text = self.preprocessor.clean(text)

        # STEP 2-6: Run analyzers (each wrapped so a single-engine failure
        # never crashes the whole feedback submission -- we fall back to 0.0).
        try:
            vader_score = self.vader.analyze(cleaned_text)["compound"]
        except Exception:
            vader_score = 0.0
        try:
            textblob_score = self.textblob.analyze(cleaned_text)["polarity"]
        except Exception:
            textblob_score = 0.0
        try:
            afinn_score = self.afinn.analyze(cleaned_text)
        except Exception:
            afinn_score = 0.0
        try:
            sentiwordnet_score = self.sentiwordnet.analyze(cleaned_text)
        except Exception:
            sentiwordnet_score = 0.0
        try:
            custom_score = self.lexicon.calculate_score(cleaned_text)
        except Exception:
            custom_score = 0.0

        # STEP 7: Emotions
        emotion_result = self.emotion.analyze(cleaned_text)

        # STEP 8: Sarcasm detection
        sarcasm = self._detect_sarcasm(text, vader_score, textblob_score)
        is_sarcastic = sarcasm["is_sarcastic"]
        sarcasm_conf = sarcasm["confidence"]

        # STEP 9: Aspect extraction
        aspects = self._extract_aspects(cleaned_text)

        # STEP 10: Intensity modifier
        intensity_mult = self._calculate_intensity(cleaned_text)

        # STEP 11: Apply sarcasm adjustment
        if is_sarcastic and sarcasm_conf > 0.5:
            vader_score = -vader_score * sarcasm_conf
            textblob_score = -textblob_score * sarcasm_conf
            afinn_score = -afinn_score * sarcasm_conf
            sentiwordnet_score = -sentiwordnet_score * sarcasm_conf
            custom_score = -custom_score * sarcasm_conf

        # STEP 12: Apply intensity
        vader_score *= intensity_mult
        textblob_score *= intensity_mult
        afinn_score *= intensity_mult
        sentiwordnet_score *= intensity_mult
        custom_score *= intensity_mult

        # STEP 13: Combine with details
        decision_details = self.decision.combine_with_details(
            vader=vader_score, textblob=textblob_score,
            afinn=afinn_score, sentiwordnet=sentiwordnet_score,
            custom=custom_score
        )
        final_score = decision_details["final_score"]

        # STEP 14: Confidence
        confidence = self.confidence.calculate([
            vader_score, textblob_score, afinn_score,
            sentiwordnet_score, custom_score
        ])

        # STEP 15: Sentiment label
        if final_score >= 0.05:
            sentiment = "Positive"
        elif final_score <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        # STEP 16: Unknown words
        unknown_words = self.unknown.detect(cleaned_text)

        return {
            "raw_text": text,
            "cleaned_text": cleaned_text,
            "sentiment": sentiment,
            "final_score": round(final_score, 4),
            "confidence": round(confidence, 2),
            "emotion": emotion_result,
            "vader_score": round(vader_score, 4),
            "textblob_score": round(textblob_score, 4),
            "afinn_score": round(afinn_score, 4),
            "sentiwordnet_score": round(sentiwordnet_score, 4),
            "custom_score": round(custom_score, 4),
            "unknown_words": unknown_words,
            "aspects": aspects,
            "is_sarcastic": is_sarcastic,
            "sarcasm_confidence": round(sarcasm_conf, 3),
            "intensity_multiplier": round(intensity_mult, 2),
            "decision_details": decision_details
        }

    def _empty_result(self, text):
        """Return empty result structure."""
        return {
            "raw_text": text, "cleaned_text": "",
            "sentiment": "Neutral", "final_score": 0.0,
            "confidence": 0.0,
            "emotion": {
                "dominant_emotion": "neutral",
                "emotion_scores": {}, "emotion_intensities": {},
                "secondary_emotions": [], "compound_mood": "neutral"
            },
            "vader_score": 0.0, "textblob_score": 0.0,
            "afinn_score": 0.0, "sentiwordnet_score": 0.0,
            "custom_score": 0.0, "unknown_words": [],
            "aspects": [], "is_sarcastic": False,
            "sarcasm_confidence": 0.0, "intensity_multiplier": 1.0,
            "decision_details": {}
        }

    def _extract_aspects(self, text: str) -> List[Dict]:
        """
        Extract aspect-based sentiment from text.
        Returns list of aspects with sentiment per aspect.
        """
        if not text:
            return []

        text_lower = text.lower()
        aspects = []

        for category, keywords in ASPECT_KEYWORDS.items():
            matched = [kw for kw in keywords if kw in text_lower]
            if matched:
                aspect_result = self._score_aspect(text_lower, matched)
                aspects.append({
                    "aspect": category,
                    "sentiment": aspect_result["label"],
                    "score": round(aspect_result["score"], 3),
                    "keywords": matched[:5],
                    "confidence": round(aspect_result["confidence"], 2)
                })

        aspects.sort(key=lambda a: abs(a["score"]), reverse=True)
        return aspects

    def _score_aspect(self, text: str, keywords: List[str]) -> Dict:
        """Score sentiment for a specific aspect using context window."""
        words = text.split()
        context_words = set()

        for i, word in enumerate(words):
            for kw in keywords:
                if kw in word:
                    start = max(0, i - 5)
                    end = min(len(words), i + 6)
                    for j in range(start, end):
                        context_words.add(words[j].strip(".,!?;:'\"()[]{}"))

        context_text = " ".join(context_words)
        custom_score = self.lexicon.calculate_score(context_text)
        vader_score = self.vader.analyze(context_text)["compound"]

        combined = custom_score * 0.6 + vader_score * 0.4

        if combined >= 0.05:
            label = "Positive"
        elif combined <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"

        return {
            "label": label,
            "score": combined,
            "confidence": min(100.0, len(context_words) * 10)
        }

    def _detect_sarcasm(
        self, text: str, vader_score: float, textblob_score: float
    ) -> Dict:
        """
        Detect sarcasm using patterns, sentiment contradiction,
        and punctuation signals.
        """
        if not text:
            return {"is_sarcastic": False, "confidence": 0.0}

        text_lower = text.lower()
        confidence = 0.0

        # Signal 1: Pattern matching
        for pattern, weight in SARACSM_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                confidence += weight

        # Signal 2: VADER/TextBlob contradiction
        if vader_score > 0.3 and textblob_score < -0.2:
            confidence += min(0.6, abs(vader_score - textblob_score))
        elif vader_score < -0.3 and textblob_score > 0.2:
            confidence += min(0.6, abs(vader_score - textblob_score))

        # Signal 3: Excessive punctuation
        excl = text.count("!")
        quest = text.count("?")
        if excl >= 3:
            confidence += min(0.4, excl * 0.1)
        if quest >= 3:
            confidence += min(0.4, quest * 0.1)

        # Signal 4: Quoted positive words
        quoted = re.findall(r'"([^"]*)"', text)
        pos_quoted = sum(1 for q in quoted if any(
            w in q.lower() for w in ["great", "excellent", "wonderful",
                                     "fantastic", "amazing", "nice",
                                     "good", "best", "perfect"]
        ))
        if pos_quoted > 0:
            confidence += min(0.5, pos_quoted * 0.25)

        # Signal 5: All caps
        caps = re.findall(r'\b[A-Z]{3,}\b', text)
        if len(caps) >= 2:
            confidence += min(0.3, len(caps) * 0.1)

        confidence = min(1.0, confidence)
        return {"is_sarcastic": confidence >= 0.4, "confidence": confidence}

    def _calculate_intensity(self, text: str) -> float:
        """
        Calculate intensity multiplier based on boosters and diminishers.
        Returns value like 1.5 for amplified or 0.5 for diminished.
        """
        if not text:
            return 1.0

        text_lower = text.lower()
        multiplier = 1.0

        for modifier, factor in INTENSITY_MODIFIERS.items():
            if modifier in text_lower:
                if factor > 1.0:
                    multiplier *= factor
                else:
                    multiplier *= factor

        # Keep within reasonable bounds
        return max(0.2, min(3.0, multiplier))
