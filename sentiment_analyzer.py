import re
from cleaning import clean_text as unified_clean_text, censor_text as unified_censor_text

# Authoritative critical-safety vocabulary shared across the whole stack.
# Guarantees kidnapping/assault/shooting/etc. are never downgraded to Neutral.
from sentiment.safety_vocabulary import (
    has_critical_safety,
    has_safety_concern,
    get_critical_terms,
    get_safety_concern_terms,
    is_discussion_context,
)

from sentiment.hybrid_engine import HybridSentimentEngine
from sentiment import context_analyzer as semantic_context

hybrid_engine = HybridSentimentEngine()

# ==================== TEXT PREPROCESSING ====================
# Unified preprocessing pipeline lives in cleaning.py.

def clean_text(text):
    """Backward-compatible alias for the unified preprocessing pipeline."""
    return unified_clean_text(text)


def has_profanity(text):
    """Detect profanity using the same neutral-replacement pipeline.

    cleaning.py replaces profanity (including obfuscated forms) with neutral alternatives.
    So we flag profanity by running the censor/detect step indirectly:
      - If the cleaned text differs from a profanity-free version is hard to infer,
        so we use the presence of any profanity by scanning known tokens is not reliable.

    Instead, we approximate by comparing cleaned output against a re-censored output.
    For compatibility, we keep it simple: if censor_text changes any profanity token,
    it indicates profanity.
    """
    if text is None:
        return False

    original = str(text)
    # cleaning.censor_text replaces profanities (including obfuscations) with neutral alternatives.
    recoded = unified_censor_text(original)
    return recoded != original.strip()


def censor_text(text):
    """Backward-compatible alias for the unified profanity handling."""
    return unified_censor_text(text)

# ==================== SENTIMENT ANALYSIS ====================


def analyze_sentiment(text):
    """Authoritative sentiment API used by feedback, chat and forum.

    This function deliberately delegates to the same HybridSentimentEngine as
    process_feedback so there is no hidden VADER-only sentiment path.
    """
    result = hybrid_engine.analyze(text)
    return result["sentiment"], result["final_score"]

def get_sentiment_explanation(text, top_n=6):
    """Return explanation evidence from the authoritative hybrid analysis."""
    if not text:
        return {"sentiment_explanation": [], "compound": 0.0, "confidence": 0.0}

    result = hybrid_engine.analyze(text)
    reasons = result.get("decision_reasons", [])
    context = result.get("context", {}) or {}
    phrases = context.get("phrases", []) or []
    explanation = []

    for phrase in phrases:
        explanation.append({"word": phrase, "impact": round(float(context.get("score", 0.0)), 3), "source": "context"})

    for reason in reasons:
        if reason not in {x.get("word") for x in explanation}:
            explanation.append({"word": reason, "impact": round(result.get("final_score", 0.0), 3), "source": "decision"})

    # Add the strongest custom-lexicon influencers for transparency.
    custom = result.get("custom_score")
    if custom is not None and abs(custom) >= 0.05:
        explanation.append({"word": "custom domain lexicon", "impact": round(custom, 3), "source": "custom"})

    return {
        "sentiment_explanation": explanation[:top_n],
        "compound": result.get("final_score", 0.0),
        "confidence": result.get("confidence", 0.0),
        "review_required": result.get("review_required", False),
        "model_version": result.get("model_version"),
    }


def build_ai_explanation(text, analysis=None, category=None, recommendation=None, max_chars=260, final_sentiment=None, final_confidence=None):
    """Create a short, student-friendly explanation from the AI analysis.

    This is intentionally generated from the structured hybrid-analysis result
    rather than hard-coded per feedback item. It keeps the mobile UI concise
    while making the classification understandable in everyday English.
    """
    if analysis is None:
        analysis = hybrid_engine.analyze(text or "")

    # When a feedback item already has a persisted final classification, that
    # result is authoritative. The explanation must describe it, not silently
    # re-classify the text a second time.
    sentiment = final_sentiment or analysis.get("sentiment", "Neutral")
    confidence = float(final_confidence if final_confidence is not None else analysis.get("confidence", 0) or 0)
    safety_mode = analysis.get("safety_mode", "none")
    context_data = analysis.get("context", {}) or {}
    reasons = analysis.get("decision_reasons", []) or []
    phrases = context_data.get("phrases", []) or []
    resolutions = context_data.get("resolutions", []) or []

    # Prefer the strongest semantic evidence over raw model scores/keywords.
    # If a persisted final sentiment was supplied, it is authoritative.
    if final_sentiment is None and safety_mode == "critical_incident":
        summary = "A serious safety or security incident was reported."
        why = "The feedback describes an event that could affect student safety."
    elif final_sentiment is None and safety_mode == "safety_concern":
        summary = "A safety or security concern was reported."
        why = "The feedback describes a situation that may affect safety."
    elif final_sentiment is None and resolutions:
        summary = "The feedback describes an issue that has been resolved or improved."
        why = "The wording indicates that the earlier problem has ended or is being successfully addressed."
    elif final_sentiment is None and any(str(r).startswith("prevention:") for r in reasons):
        summary = "The feedback describes a safety issue being prevented or successfully handled."
        why = "The AI considered the action taken, not just the word describing the incident."
    elif sentiment == "Negative":
        summary = "The feedback describes a problem or unsatisfactory experience."
        why = "The wording indicates that something is not working well or has caused a concern."
    elif sentiment == "Positive":
        summary = "The feedback describes a positive experience or improvement."
        why = "The wording shows satisfaction, appreciation, or a successful outcome."
    else:
        summary = "The feedback does not clearly express a positive or negative experience."
        why = "The wording is mainly factual, balanced, or not strong enough for a confident sentiment decision."

    if category:
        category_label = str(category).replace("_", " ").replace("ict/wifi", "ICT/Wi-Fi")
        summary = f"{summary} Category: {category_label}."

    if confidence and confidence < 55:
        why += " The AI is not fully confident, so an admin review may be needed."

    recommendation_text = ""
    if recommendation:
        rec = recommendation.get("short_term_solution") if isinstance(recommendation, dict) else None
        if rec:
            recommendation_text = str(rec).strip()
    if not recommendation_text:
        if final_sentiment is None and safety_mode == "critical_incident":
            recommendation_text = "Recommended action: review this report promptly."
        elif sentiment == "Negative":
            recommendation_text = "Recommended action: review the issue and follow up."
        elif sentiment == "Positive":
            recommendation_text = "Recommended action: no urgent action is required."
        else:
            recommendation_text = "Recommended action: review the feedback if clarification is needed."

    # Keep each field short enough for a phone without hiding the meaning.
    def clip(value, limit):
        value = re.sub(r"\s+", " ", str(value or "")).strip()
        return value if len(value) <= limit else value[:limit - 1].rstrip(" ,;:") + "…"

    return {
        "summary": clip(summary, max_chars),
        "why": clip(why, max_chars),
        "recommendation": clip(recommendation_text, max_chars),
        "confidence": round(confidence),
        "model_version": analysis.get("model_version"),
        "review_required": bool(analysis.get("review_required", False)),
    }

def calculate_urgency(text, sentiment):
    """Calculate urgency using context-aware phrases and separate safety logic."""
    text = str(text or "")
    normalized, _ = semantic_context.normalize_domain_language(text)
    context = semantic_context.analyze_context(normalized)

    # Critical safety keywords always get maximum urgency
    if has_critical_safety(normalized):
        return 5

    # Check for serious safety keywords even in discussion context
    # These topics are serious enough to warrant attention
    serious_safety_terms = [
        # Violent crimes
        "kidnap", "kidnapped", "kidnapping", "abduct", "abduction", "hostage",
        "shooting", "gunshot", "gunshots", "weapon", "murder", "rape", "sexual assault",
        "assault", "stabbing", "bomb", "explosion", "terrorist", "terrorism",
        "attacked", "attacker", "armed", "robbery", "robbed", "mugged", "mugging",
        # Threats & harassment
        "threat", "threatened", "threatening", "stalker", "stalking", "harassment",
        "harassed", "intimidated", "intimidation", "blackmail", "extortion",
        # Health emergencies
        "death", "died", "fatal", "suicide", "self harm", "overdose", "unconscious",
        "seizure", "heart attack", "stroke", "ambulance", "hospitalized",
        # Severe incidents
        "riot", "riotous", "mob", "lynching", "arson", "sabotage", "vandalism",
        "break in", "burglary", "theft", "stolen", "intruder", "trespassing",
    ]
    has_serious_safety = any(term in normalized.lower() for term in serious_safety_terms)

    # Resolved/no-longer language suppresses the raw severity of nouns such as
    # "flood" when the problem has explicitly ended.
    resolution = bool(context.get("resolutions"))
    discussion = is_discussion_context(normalized)

    urgency = 1
    patterns = {
        5: [
            # Critical safety & emergencies
            "emergency", "danger", "injured", "unsafe", "hazard", "assault", "harassment",
            "urgent", "critical", "gunshots", "gunshot", "shooting", "weapon", "violence",
            "hostage", "threat", "armed", "attack", "blood", "injury", "panic", "fight",
            "fire outbreak", "building fire", "explosion", "bomb", "terrorist", "murder",
            "rape", "sexual assault", "stabbing", "kidnapping", "kidnapped", "abduction",
            # Health emergencies
            "death", "died", "fatal", "life threatening", "medical emergency", "ambulance",
            "hospitalized", "unconscious", "seizure", "overdose", "suicide", "self harm",
            # Severe infrastructure
            "building collapse", "structural damage", "gas leak", "electrocution",
            "trapped", "stuck in elevator", "elevator failure",
        ],
        4: [
            # Infrastructure failures
            "still not working", "not working", "broken", "no water", "no power",
            "no electricity", "collapsed", "fire", "ignored", "weeks", "out of order",
            "not functional", "malfunctioning", "down", "offline", "disconnected",
            # Health & safety concerns
            "sick", "illness", "contagious", "infection", "infestation", "pest",
            "cockroach", "rat", "mold", "asbestos", "chemical spill",
            # Security concerns
            "theft", "stolen", "robbery", "break in", "burglary", "vandalism",
            "suspicious person", "intruder", "stalker", "threatening",
            # Academic concerns
            "failed", "failing", "academic dismissal", "suspension", "expulsion",
            "wrong grade", "missing results", "not credited",
            # Service failures
            "no internet", "wifi down", "portal down", "system crash",
            "payment not reflected", "no refund",
        ],
        3: [
            # Delays & inefficiencies
            "delay", "late", "rude", "unhelpful", "expensive", "overcharged",
            "frustrated", "annoying", "slow", "long wait", "queue", "waiting",
            "postponed", "rescheduled", "cancelled", "backlog",
            # Academic issues
            "unclear", "confusing", "disorganized", "unfair", "biased",
            "inconsistent", "outdated", "irrelevant", "boring", "monotone",
            "absent lecturer", "missed class", "no substitute",
            # Facilities issues
            "dirty", "unclean", "unsanitary", "smelly", "foul odor",
            "leaking", "flooded", "clogged", "broken furniture",
            "no lighting", "dark", "poor ventilation", "too hot", "too cold",
            # Service quality
            "poor quality", "bad service", "unprofessional", "unresponsive",
            "no feedback", "ignored complaint", "no action",
            # Financial
            "overpriced", "hidden charges", "unexpected fee", "expensive",
            # Accommodation
            "noisy", "loud", "thin walls", "no privacy", "overcrowded",
            "no hot water", "cold water", "bed bugs", "mosquitoes",
        ],
        2: [
            # Minor inconveniences
            "small", "noisy", "crowded", "uncomfortable", "inconvenient",
            "far", "distance", "walking", "parking", "space",
            # Suggestions for improvement
            "could be better", "needs improvement", "upgrade", "renovation",
            "more resources", "additional", "expand", "extend hours",
            # Mild preferences
            "prefer", "wish", "hope", "suggest", "recommend",
            "would be nice", "better if", "should add",
            # Minor issues
            "minor", "slight", "occasionally", "sometimes", "rarely",
            "not always", "a bit", "somewhat",
        ],
    }

    for level, terms in patterns.items():
        for term in terms:
            if re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", normalized):
                if discussion and term in get_critical_terms() + get_safety_concern_terms():
                    continue
                urgency = max(urgency, level)

    if resolution:
        # Explicit resolution/prevention language describes a problem that has
        # ended or been successfully handled; do not leave incident-level
        # urgency behind just because a severe noun appears in the sentence.
        urgency = min(urgency, 2)

    # Boost urgency for serious safety topics even in discussion context
    if has_serious_safety and urgency < 3:
        urgency = 3

    if sentiment == "Negative" and urgency < 3 and not resolution:
        urgency = 3
    
    # Cap urgency for positive and neutral feedback
    if sentiment == "Positive":
        urgency = min(urgency, 2)
    elif sentiment == "Neutral":
        urgency = min(urgency, 3)

    return min(urgency, 5)

def get_urgency_explanation(text, sentiment):
    text = str(text or "")
    normalized, _ = semantic_context.normalize_domain_language(text)
    if has_critical_safety(normalized):
        terms = [t for t in get_critical_terms() if re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", normalized, re.I)]
        return {"urgency_keywords_detected": terms[:12], "sentiment_boost_applied": False,
                "base_urgency_from_keywords": 5, "final_urgency": 5, "sentiment": sentiment,
                "critical_safety_override": True}

    context = semantic_context.analyze_context(normalized)
    urgency = calculate_urgency(normalized, sentiment)
    detected = list(context.get("phrases") or [])
    if context.get("resolutions"):
        detected.extend(context["resolutions"])
    return {
        "urgency_keywords_detected": list(dict.fromkeys(detected))[:12],
        "sentiment_boost_applied": sentiment == "Negative" and urgency >= 3 and not context.get("resolutions"),
        "base_urgency_from_keywords": urgency,
        "final_urgency": urgency,
        "sentiment": sentiment,
        "critical_safety_override": False,
        "resolution_context": bool(context.get("resolutions")),
    }

def detect_category(text):
    text_lower = text.lower()

    # A genuine critical safety incident always maps to the Safety category,
    # regardless of any incidental keywords from other categories.
    if has_critical_safety(str(text)):
        return 'Safety'

    discussion = is_discussion_context(str(text))
    safety_vocab_terms = set(get_critical_terms()) | set(get_safety_concern_terms())

    category_keywords = {
        'Accommodation': ['hostel', 'dorm', 'room', 'bed', 'water', 'toilet', 'shower', 'accommodation', 'hall', 'bathroom', 'flood', 'leak'],
        'ICT/Wi-Fi': ['wifi', 'internet', 'network', 'connection', 'computer', 'lab', 'portal', 'slow', 'disconnect', 'server'],
        'Academics': ['lecturer', 'class', 'exam', 'examination', 'malpractice', 'course', 'assignment', 'timetable', 'curriculum', 'grade', 'result', 'lecture', 'test', 'quiz', 'project', 'dissertation', 'thesis', 'research', 'study', 'academic', 'semester', 'credit', 'gpa', 'cgpa'],
        'Catering': ['food', 'canteen', 'cafeteria', 'meal', 'dining', 'hungry', 'price', 'restaurant', 'kitchen', 'cook'],
        'Facilities': ['library', 'classroom', 'building', 'elevator', 'light', 'fan', 'ac', 'chair', 'desk', 'projector', 'flood', 'flooding', 'leak', 'leaking'],
        'Safety': ['security', 'safe', 'theft', 'dark', 'lighting', 'patrol', 'gate', 'danger', 'unsafe', 'cctv',
                     'gunshots', 'gunshot', 'shooting', 'weapon', 'violence', 'assault', 'harassment', 'emergency',
                     'threat', 'threatened', 'hostage', 'armed', 'attack', 'fight', 'blood', 'injured', 'injury',
                     'panic', 'hazard', 'unsafe'] + get_critical_terms() + get_safety_concern_terms(),
        'Transport': ['bus', 'shuttle', 'parking', 'transport', 'vehicle', 'car', 'driver', 'fuel', 'trotro'],
        'Mental Health': ['stress', 'anxiety', 'counseling', 'mental', 'wellness', 'support', 'depression', 'pressure']
    }
    
    scores = {cat: 0 for cat in category_keywords}
    for category, keywords in category_keywords.items():
        for keyword in keywords:
            if category == 'Safety' and discussion and keyword in safety_vocab_terms:
                continue
            if keyword in text_lower:
                scores[category] += 1
    
    if max(scores.values()) > 0:
        best = max(scores, key=scores.get)
        # A discussion/teaching context mentioning safety vocabulary is not
        # itself an active safety report. Route it to Other unless the text
        # contains explicit incident language that already triggered the
        # safety override above.
        if best == 'Safety' and discussion:
            return 'Other'
        return best
    return 'Other'

# ==================== MAIN PROCESSING ====================

def process_feedback(text, user_category=None):

    # =====================================
    # 1. Run the Hybrid Sentiment Engine
    # =====================================
    # This can fail if NLTK language data isn't available yet (e.g. a cold
    # start on Render where the background NLTK download hasn't finished).
    # A sentiment-analysis failure must never cost a student their feedback
    # submission, so we fall back to a safe neutral result instead of
    # letting the exception propagate into a 500 error.
    try:
        hybrid_result = hybrid_engine.analyze(text)
    except Exception as e:
        print(f"[sentiment_analyzer] hybrid_engine.analyze failed, falling back to neutral: {e}")
        hybrid_result = {
            "cleaned_text": text,
            "sentiment": "Neutral",
            "final_score": 0.0,
            "confidence": 0.0,
            "emotion": {
                "dominant_emotion": "neutral",
                "emotion_scores": {},
                "emotion_intensities": {},
                "secondary_emotions": [],
                "compound_mood": "neutral",
            },
            "unknown_words": [],
        }


    # =====================================
    # 2. Get results from the hybrid engine
    # =====================================

    cleaned_text = hybrid_result["cleaned_text"]

    sentiment = hybrid_result["sentiment"]

    score = hybrid_result["final_score"]


    # =====================================
    # 3. Run the existing profanity check
    # =====================================

    profanity_flag = has_profanity(text)


    # =====================================
    # 4. Calculate urgency
    # =====================================

    urgency = calculate_urgency(
        cleaned_text,
        sentiment
    )


    # =====================================
    # 5. Detect or use selected category
    # =====================================

    if user_category and user_category != "Other":

        category = user_category

    else:

        category = detect_category(
            cleaned_text
        )


    # =====================================
    # 6. Return results to app.py
    # =====================================

    return {

        "sentiment": sentiment,

        "sentiment_score": round(
            score,
            3
        ),

        "urgency_score": urgency,

        "detected_category": category,

        "cleaned_text": cleaned_text,

        "has_profanity": profanity_flag,

        # Extra hybrid results
        "confidence": hybrid_result[
            "confidence"
        ],

        "emotion": hybrid_result[
            "emotion"
        ],

        "unknown_words": hybrid_result[
            "unknown_words"
        ]

    }

def analyze_chat_message(message):
    result = hybrid_engine.analyze(message)
    sentiment = result["sentiment"]
    urgency = calculate_urgency(result.get("normalized_text") or result.get("cleaned_text") or message, sentiment)
    is_flagged = (sentiment == "Negative" and urgency >= 3) or result.get("safety_mode") == "critical_incident"
    return {
        "sentiment": sentiment,
        "sentiment_score": round(result["final_score"], 3),
        "urgency_score": urgency,
        "cleaned_message": result["cleaned_text"],
        "is_flagged": is_flagged,
        "confidence": result.get("confidence", 0.0),
        "review_required": result.get("review_required", False),
        "model_version": result.get("model_version"),
    }

def analyze_topic(content, replies=[]):
    main = hybrid_engine.analyze(content)
    main_sentiment = main["sentiment"]
    main_score = main["final_score"]
    main_urgency = calculate_urgency(main.get("normalized_text") or content, main_sentiment)

    reply_results = []
    for reply in replies:
        reply_text = reply.content if hasattr(reply, "content") else str(reply)
        r = hybrid_engine.analyze(reply_text)
        reply_results.append(r)

    all_results = [main] + reply_results
    positive = sum(1 for r in all_results if r["sentiment"] == "Positive")
    negative = sum(1 for r in all_results if r["sentiment"] == "Negative")
    if positive > negative:
        topic_sentiment = "Positive"
    elif negative > positive:
        topic_sentiment = "Negative"
    else:
        topic_sentiment = "Neutral"

    scores = [main_score * 2] + [r["final_score"] for r in reply_results]
    weighted_score = sum(scores) / (2 + len(reply_results)) if reply_results else main_score
    max_urgency = max([main_urgency] + [calculate_urgency(r.get("normalized_text", ""), r["sentiment"]) for r in reply_results])
    avg_confidence = sum(r.get("confidence", 0.0) for r in all_results) / len(all_results)

    return {
        "sentiment": topic_sentiment,
        "sentiment_score": round(weighted_score, 3),
        "urgency_score": max_urgency,
        "confidence": round(avg_confidence, 2),
        "review_required": any(r.get("review_required") for r in all_results),
        "model_version": "HTU-Sentiment-v3-context-fusion",
    }

def get_room_sentiment_summary(messages):
    if not messages:
        return {'total': 0, 'positive': 0, 'negative': 0, 'neutral': 0, 'avg_urgency': 0, 'sentiment_score': 0}
    
    total = len(messages)
    positive = sum(1 for m in messages if m.sentiment == 'Positive')
    negative = sum(1 for m in messages if m.sentiment == 'Negative')
    neutral = total - positive - negative
    avg_urgency = sum(m.urgency_score for m in messages) / total if total > 0 else 0
    sentiment_score = (positive - negative) / total if total > 0 else 0
    
    return {
        'total': total, 'positive': positive, 'negative': negative, 'neutral': neutral,
        'positive_pct': round(positive / total * 100, 1) if total > 0 else 0,
        'negative_pct': round(negative / total * 100, 1) if total > 0 else 0,
        'avg_urgency': round(avg_urgency, 1),
        'sentiment_score': round(sentiment_score, 3)
    }

def get_forum_sentiment_summary(topics):
    if not topics:
        return {'total_topics': 0, 'total_replies': 0, 'positive_topics': 0, 'negative_topics': 0,
                'neutral_topics': 0, 'avg_sentiment': 0, 'hot_topics': 0, 'top_categories': {}}
    
    total_topics = len(topics)
    positive_topics = sum(1 for t in topics if t.sentiment == 'Positive')
    negative_topics = sum(1 for t in topics if t.sentiment == 'Negative')
    neutral_topics = total_topics - positive_topics - negative_topics
    total_replies = sum(len(t.replies) for t in topics)
    hot_topics = sum(1 for t in topics if len(t.replies) > 10 or t.urgency_score >= 4)
    avg_score = sum(t.sentiment_score for t in topics if t.sentiment_score) / total_topics if total_topics > 0 else 0
    
    category_counts = {}
    for t in topics:
        category_counts[t.category] = category_counts.get(t.category, 0) + 1
    
    return {
        'total_topics': total_topics, 'total_replies': total_replies,
        'positive_topics': positive_topics, 'negative_topics': negative_topics, 'neutral_topics': neutral_topics,
        'positive_pct': round(positive_topics / total_topics * 100, 1) if total_topics > 0 else 0,
        'negative_pct': round(negative_topics / total_topics * 100, 1) if total_topics > 0 else 0,
        'avg_sentiment': round(avg_score, 3),
        'hot_topics': hot_topics,
        'top_categories': sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    }