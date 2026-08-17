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
from sentiment.safety_vocabulary import (has_critical_safety, has_safety_concern, is_discussion_context,
                                            get_critical_terms, get_safety_concern_terms)
from sentiment import context_analyzer as context


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
        "security", "safe", "theft", "stolen", "robbery", "robbed", "dark",
        "lighting", "patrol", "gate", "guard", "cctv", "camera",
        "danger", "unsafe", "gunshots", "shooting", "weapon", "violence",
        "assault", "assaulted", "harassment", "emergency", "threat", "threatened",
        "attack", "attacked", "fight", "blood", "injured", "injury", "panic",
        "hazard", "intruder", "suspicious", "vandalism", "fire",
        "kidnap", "kidnapped", "kidnapping", "kidnapper", "abduct", "abducted",
        "abduction", "hostage", "stab", "stabbed", "stabbing", "rape", "raped",
        "bomb", "explosion", "explode", "shooter", "gun", "firearm"
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
        self.context = context

    def analyze(self, text: str) -> dict:
        """Run the unified, context-aware sentiment pipeline.

        The pipeline is deliberately fault tolerant: unavailable third-party
        engines are represented as ``None`` and are excluded from fusion rather
        than being converted to fake Neutral evidence.
        """
        if not text or not str(text).strip():
            return self._empty_result(text)

        raw_text = str(text)
        cleaned_text = self.preprocessor.clean(raw_text)

        # Domain/context layer runs first so generic models can be interpreted
        # in light of negation, resolution, complaint phrases and local slang.
        context = self.context.analyze_context(raw_text)
        normalized_context_text, slang = self.context.normalize_domain_language(cleaned_text)
        model_text = normalized_context_text or cleaned_text

        scores = {}
        engine_details = {}

        try:
            result = self.vader.analyze(model_text)
            scores["vader"] = result["compound"] if result else None
            engine_details["vader"] = result
        except Exception as exc:
            scores["vader"] = None
            engine_details["vader_error"] = str(exc)

        try:
            result = self.textblob.analyze(model_text)
            scores["textblob"] = result["polarity"] if result else None
            engine_details["textblob"] = result
        except Exception as exc:
            scores["textblob"] = None
            engine_details["textblob_error"] = str(exc)

        try:
            result = self.afinn.analyze(model_text)
            scores["afinn"] = result
            engine_details["afinn"] = result
        except Exception as exc:
            scores["afinn"] = None
            engine_details["afinn_error"] = str(exc)

        try:
            result = self.sentiwordnet.analyze(model_text)
            # SentiWordNet's old 0.0 fallback means "unavailable" in some
            # deployments. Treat it as missing when the adapter says so.
            scores["sentiwordnet"] = result if result is not None else None
            engine_details["sentiwordnet"] = result
        except Exception as exc:
            scores["sentiwordnet"] = None
            engine_details["sentiwordnet_error"] = str(exc)

        try:
            scores["custom"] = self.lexicon.calculate_score(model_text)
            engine_details["custom"] = scores["custom"]
        except Exception as exc:
            scores["custom"] = None
            engine_details["custom_error"] = str(exc)

        # Emotion detection is supplementary evidence, not the sentiment
        # classifier itself.
        try:
            emotion_result = self.emotion.analyze(model_text)
        except Exception:
            emotion_result = {
                "dominant_emotion": "neutral",
                "emotion_scores": {},
                "emotion_intensities": {},
                "secondary_emotions": [],
                "compound_mood": "neutral",
            }

        # Sarcasm and intensity are kept, but are prevented from being applied
        # blindly to every model. Context phrases remain authoritative.
        vader_score = scores.get("vader") or 0.0
        textblob_score = scores.get("textblob") or 0.0
        sarcasm = self._detect_sarcasm(raw_text, vader_score, textblob_score)
        intensity_mult = self._calculate_intensity(model_text)

        for key in ("vader", "textblob", "afinn", "sentiwordnet", "custom"):
            if scores.get(key) is not None:
                scores[key] = max(-1.0, min(1.0, scores[key] * intensity_mult))

        # Do not let weak sarcasm flip a clear domain phrase.
        if sarcasm["is_sarcastic"] and sarcasm["confidence"] >= 0.65 and abs(context.get("score", 0.0)) < 0.45:
            for key in ("vader", "textblob", "afinn", "sentiwordnet", "custom"):
                if scores.get(key) is not None:
                    scores[key] *= -1

        context_label = self.context.sentiment_from_context(context)
        decision_details = self.decision.combine_named(
            scores,
            context_score=context.get("score", 0.0),
            context_confidence=context.get("confidence", 0.0),
            context_label=context_label,
        )
        final_score = decision_details["final_score"]

        # Critical safety is a separate classifier. It cannot be diluted by a
        # generic sentiment ensemble. Discussion/prevention context is handled
        # inside the authoritative safety vocabulary module.
        critical_safety_hit = has_critical_safety(model_text)
        safety_concern_hit = has_safety_concern(model_text)
        discussion_context = is_discussion_context(model_text)
        if critical_safety_hit:
            final_score = -0.90
            sentiment = "Negative"
            safety_mode = "critical_incident"
        elif safety_concern_hit:
            final_score = min(final_score, -0.30)
            sentiment = "Negative"
            safety_mode = "safety_concern"
        elif context_label in {"Positive", "Negative"} and context.get("confidence", 0) >= 0.45:
            sentiment = context_label if abs(context.get("score", 0)) >= 0.15 else self._label(final_score)
            if context_label == "Negative":
                final_score = min(final_score, -0.12)
            elif context_label == "Positive":
                final_score = max(final_score, 0.12)
            safety_mode = "context"
        else:
            sentiment = self._label(final_score)
            safety_mode = "none"

        # A weak positive custom-lexicon hit on its own (e.g. "opens",
        # "fine", "safe") is not enough to call otherwise neutral factual
        # feedback Positive. Strong positive wording or an independent model
        # can still produce Positive.
        custom_only_positive = (
            sentiment == "Positive"
            and context_label is None
            and (scores.get("custom") or 0.0) < 0.45
            and all((v is None or v <= 0.10) for k, v in scores.items() if k != "custom")
        )
        if custom_only_positive:
            final_score = 0.0
            sentiment = "Neutral"

        if "context:qualified_neutral" in (context.get("reasons") or []):
            final_score = 0.0
            sentiment = "Neutral"

        # Discussion language should not turn a safety topic into an incident.
        # If the only positive signal came from generic words such as "workshop"
        # or "safe", classify the safety topic itself as Neutral. Explicit
        # negative wording (e.g. "waste of time") is preserved.
        if discussion_context and not critical_safety_hit and not safety_concern_hit:
            safety_mode = "discussion_only"
            safety_terms = set(get_critical_terms()) | set(get_safety_concern_terms())
            has_safety_topic = any(re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", model_text, re.I) for term in safety_terms)
            generic_model_score = scores.get("textblob")
            custom_score_now = scores.get("custom") or 0.0
            # Positive custom scores can come from words such as "safe" or
            # "security" themselves. They are not enough to call a discussion
            # positive. Require independent positive model/context evidence.
            if has_safety_topic and not context_label and generic_model_score is not None and generic_model_score <= 0.10 and -0.15 < custom_score_now < 0.45:
                final_score = 0.0
                sentiment = "Neutral"

        # Confidence is based on actual evidence availability, context strength,
        # model agreement and proximity to the decision boundary.
        confidence = self._calculate_confidence(scores, context, final_score, sentiment, critical_safety_hit)
        review_required = confidence < 55.0 and not critical_safety_hit
        if discussion_context and not context_label and abs(final_score) < 0.15:
            review_required = False
            confidence = max(confidence, 62.0)

        aspects = self._extract_aspects(model_text)
        unknown_words = self.unknown.detect(model_text)

        available = [k for k, v in scores.items() if v is not None]
        unavailable = [k for k, v in scores.items() if v is None]
        reasons = list(context.get("reasons") or [])
        if critical_safety_hit:
            reasons.append("safety:critical_incident_override")
        elif safety_concern_hit:
            reasons.append("safety:concern_override")
        if unavailable:
            reasons.append("system:unavailable_engines_excluded")
        if review_required:
            reasons.append("confidence:human_review_recommended")

        return {
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "normalized_text": model_text,
            "sentiment": sentiment,
            "final_score": round(final_score, 4),
            "confidence": round(confidence, 2),
            "review_required": review_required,
            "model_version": "HTU-Sentiment-v3-context-fusion",
            "emotion": emotion_result,
            "vader_score": self._round_or_none(scores.get("vader")),
            "textblob_score": self._round_or_none(scores.get("textblob")),
            "afinn_score": self._round_or_none(scores.get("afinn")),
            "sentiwordnet_score": self._round_or_none(scores.get("sentiwordnet")),
            "custom_score": self._round_or_none(scores.get("custom")),
            "unknown_words": unknown_words,
            "aspects": aspects,
            "is_sarcastic": sarcasm["is_sarcastic"],
            "sarcasm_confidence": round(sarcasm["confidence"], 3),
            "intensity_multiplier": round(intensity_mult, 2),
            "context": context,
            "engine_details": engine_details,
            "available_engines": available,
            "unavailable_engines": unavailable,
            "safety_mode": safety_mode,
            "discussion_context": discussion_context,
            "decision_details": decision_details,
            "decision_reasons": reasons,
            "domain_normalizations": slang,
        }

    @staticmethod
    def _label(score: float) -> str:
        if score >= 0.05:
            return "Positive"
        if score <= -0.05:
            return "Negative"
        return "Neutral"

    @staticmethod
    def _round_or_none(value):
        return round(value, 4) if value is not None else None

    def _calculate_confidence(self, scores, context, final_score, sentiment, critical=False):
        if critical:
            return 99.0
        available = [v for v in scores.values() if v is not None]
        if not available:
            base = 35.0
        else:
            mean = sum(available) / len(available)
            variance = sum((x - mean) ** 2 for x in available) / len(available)
            agreement = max(0.0, 1.0 - min(1.0, variance ** 0.5))
            availability = min(1.0, len(available) / 5.0)
            base = 38.0 + agreement * 28.0 + availability * 18.0
        context_conf = float(context.get("confidence", 0.0) or 0.0)
        context_score = abs(float(context.get("score", 0.0) or 0.0))
        if context_conf >= 0.45:
            base += min(18.0, context_conf * 12.0 + context_score * 8.0)
        if abs(final_score) < 0.08:
            base -= 15.0
        if sentiment == "Neutral" and context_score >= 0.15:
            base -= 10.0
        return round(max(0.0, min(99.0, base)), 2)

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
            matched = []
            for kw in keywords:
                pattern = r"(?<!\w)" + re.escape(kw) + r"(?!\w)"
                if re.search(pattern, text_lower):
                    matched.append(kw)
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
        vader_result = self.vader.analyze(context_text)
        vader_score = vader_result["compound"] if vader_result else 0.0

        # If VADER is unavailable, custom/domain evidence remains valid.
        combined = custom_score * 0.7 + vader_score * 0.3

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

        for modifier, factor in sorted(INTENSITY_MODIFIERS.items(), key=lambda x: len(x[0]), reverse=True):
            pattern = r"(?<!\w)" + re.escape(modifier) + r"(?!\w)"
            if re.search(pattern, text_lower):
                multiplier *= factor

        # Do not allow multiple generic boosters to explode the score.
        return max(0.5, min(2.5, multiplier))
