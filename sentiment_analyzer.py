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

from sentiment import context_analyzer as semantic_context

# NOTE: hybrid_engine is no longer used - replaced by unified VADER + custom lexicon approach
# from sentiment.hybrid_engine import HybridSentimentEngine
# hybrid_engine = HybridSentimentEngine()

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
    """Fast, reliable sentiment analysis using VADER."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from sentiment.custom_lexicon import CustomLexiconManager
    
    analyzer = SentimentIntensityAnalyzer()
    custom_lexicon = CustomLexiconManager()
    for word, score in custom_lexicon.get_lexicon().items():
        analyzer.lexicon[word] = score
    
    vs = analyzer.polarity_scores(text)
    compound = vs['compound']
    
    if compound >= 0.05:
        sentiment = 'Positive'
    elif compound <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    
    return sentiment, round(compound, 3)

def get_sentiment_explanation(text, top_n=6):
    """Return explanation evidence using the new analyzer."""
    if not text:
        return {"sentiment_explanation": [], "compound": 0.0, "confidence": 0.0}

    # Use the new analyzer for consistent results
    explanation = []
    
    # Get custom lexicon matches
    from sentiment.custom_lexicon import CustomLexiconManager
    custom = CustomLexiconManager()
    text_lower = text.lower().strip()
    text_underscores = text_lower.replace(' ', '_')
    
    # Find matching phrases
    for phrase, score in custom.lexicon.items():
        if '_' in phrase and phrase in text_underscores:
            explanation.append({
                "word": phrase.replace('_', ' '),
                "impact": round(score, 3),
                "source": "custom"
            })
    
    # Find matching words
    words = text_lower.split()
    for word in words:
        clean_word = word.strip('.,!?;:\"\'()[]{}')
        if clean_word in custom.lexicon and abs(custom.lexicon[clean_word]) >= 0.3:
            explanation.append({
                "word": clean_word,
                "impact": round(custom.lexicon[clean_word], 3),
                "source": "custom"
            })
    
    # Sort by impact and limit
    explanation.sort(key=lambda x: abs(x['impact']), reverse=True)
    explanation = explanation[:top_n]
    
    # Calculate overall score
    sentiment, score, confidence = _analyze_sentiment_internal(text)
    
    return {
        "sentiment_explanation": explanation,
        "compound": score,
        "confidence": confidence,
    }


def _analyze_sentiment_internal(text):
    """Internal sentiment analysis using VADER + custom lexicon"""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from sentiment.custom_lexicon import CustomLexiconManager

    analyzer = SentimentIntensityAnalyzer()
    custom_lexicon = CustomLexiconManager()
    
    for word, score in custom_lexicon.lexicon.items():
        analyzer.lexicon[word] = score
    
    vs = analyzer.polarity_scores(text)
    compound = vs['compound']
    custom_score = custom_lexicon.calculate_score(text)
    
    # Combine scores (same logic as process_feedback)
    if abs(compound - custom_score) > 0.3:
        compound = custom_score
    elif abs(compound) < 0.05 and abs(custom_score) >= 0.05:
        compound = custom_score
    elif compound > 0.05 and custom_score < -0.05:
        compound = custom_score
    elif compound < -0.05 and custom_score > 0.05:
        compound = custom_score
    
    if compound >= 0.05:
        sentiment = 'Positive'
    elif compound <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    
    confidence = min(100, abs(compound) * 100 + 50)
    return sentiment, round(compound, 3), confidence


def build_ai_explanation(text, analysis=None, category=None, recommendation=None, max_chars=260, final_sentiment=None, final_confidence=None):
    """Create a short, student-friendly explanation from the AI analysis.

    Uses the new unified analyzer for consistent results across the system.
    """
    if analysis is None:
        # Use the new analyzer instead of hybrid_engine
        sentiment, score, confidence = _analyze_sentiment_internal(text or "")
        analysis = {
            "sentiment": sentiment,
            "final_score": score,
            "confidence": confidence,
            "safety_mode": "none",
            "context": {},
            "decision_reasons": [],
            "model_version": "HTU-Sentiment-v4-unified",
            "review_required": False,
        }

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

    if has_critical_safety(normalized):
        return 5

    # Resolved/no-longer language suppresses the raw severity of nouns such as
    # "flood" when the problem has explicitly ended.
    resolution = bool(context.get("resolutions"))
    discussion = is_discussion_context(normalized)

    urgency = 1
    patterns = {
        5: ["emergency", "danger", "injured", "unsafe", "hazard", "assault", "harassment", "urgent", "critical",
            "gunshots", "gunshot", "shooting", "weapon", "violence", "hostage", "threat", "armed", "attack",
            "blood", "injury", "panic", "fight", "fire outbreak"],
        4: ["still not working", "not working", "broken", "no water", "no power", "no electricity", "collapsed", "fire", "ignored", "weeks"],
        3: ["delay", "late", "rude", "unhelpful", "expensive", "overcharged", "frustrated", "annoying"],
        2: ["slow", "small", "noisy", "crowded", "uncomfortable"],
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
        'Academics': ['lecturer', 'class', 'exam', 'course', 'assignment', 'timetable', 'curriculum', 'grade', 'result', 'lecture'],
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
    """Simplified, reliable sentiment analysis using VADER + custom lexicon."""

    # 1. Clean the text
    cleaned_text = unified_clean_text(text)
    
    # 2. Run VADER sentiment analysis (fast and reliable)
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from sentiment.custom_lexicon import CustomLexiconManager
    
    analyzer = SentimentIntensityAnalyzer()
    custom_lexicon = CustomLexiconManager()
    
    # Add custom lexicon words to VADER
    for word, score in custom_lexicon.lexicon.items():
        analyzer.lexicon[word] = score
    
    vs = analyzer.polarity_scores(cleaned_text)
    compound = vs['compound']
    
    # 3. Also calculate custom lexicon score for phrase matching
    custom_score = custom_lexicon.calculate_score(cleaned_text)
    
    # 4. Combine scores - prioritize custom lexicon for sarcasm/phrase detection
    # If VADER and custom disagree significantly, trust custom (it detects phrases)
    if abs(compound - custom_score) > 0.3:
        # Significant disagreement - trust custom lexicon (phrase-level analysis)
        compound = custom_score
    elif abs(compound) < 0.05 and abs(custom_score) >= 0.05:
        # VADER is neutral but custom detects sentiment
        compound = custom_score
    elif compound > 0.05 and custom_score < -0.05:
        # VADER positive, custom negative - sarcasm detected
        compound = custom_score
    elif compound < -0.05 and custom_score > 0.05:
        # VADER negative, custom positive - understatement detected
        compound = custom_score
    
    # 5. Determine sentiment
    if compound >= 0.05:
        sentiment = 'Positive'
    elif compound <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    
    # 6. Calculate urgency
    urgency = calculate_urgency(cleaned_text, sentiment)
    
    # 7. Detect category
    if user_category and user_category != 'Other':
        category = user_category
    else:
        category = detect_category(cleaned_text)
    
    # 8. Check profanity
    profanity_flag = has_profanity(text)
    
    # 9. Calculate confidence based on compound score
    confidence = min(100, abs(compound) * 100 + 50)
    
    # 10. Simple emotion detection
    emotion_data = {
        'dominant_emotion': 'positive' if compound > 0.05 else ('negative' if compound < -0.05 else 'neutral'),
        'emotion_scores': {'positive': max(0, compound), 'negative': max(0, -compound)},
        'compound_mood': 'positive' if compound > 0.05 else ('negative' if compound < -0.05 else 'neutral'),
    }
    
    return {
        'sentiment': sentiment,
        'sentiment_score': round(compound, 3),
        'urgency_score': urgency,
        'detected_category': category,
        'cleaned_text': cleaned_text,
        'has_profanity': profanity_flag,
        'confidence': round(confidence, 1),
        'emotion': emotion_data,
        'unknown_words': [],
    }

def analyze_chat_message(message):
    """Fast sentiment analysis for chat messages."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(message)
    compound = vs['compound']
    
    if compound >= 0.05:
        sentiment = 'Positive'
    elif compound <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    
    return sentiment, round(compound, 3)


def analyze_topic(content, replies=[]):
    """Analyze topic sentiment using the new unified analyzer."""
    # Use the new analyzer for consistent results
    main_sentiment, main_score, main_confidence = _analyze_sentiment_internal(content)
    main_urgency = calculate_urgency(content, main_sentiment)

    reply_results = []
    for reply in replies:
        reply_text = reply.content if hasattr(reply, "content") else str(reply)
        r_sentiment, r_score, r_confidence = _analyze_sentiment_internal(reply_text)
        reply_results.append({
            'sentiment': r_sentiment,
            'score': r_score,
            'confidence': r_confidence,
        })

    all_results = [{'sentiment': main_sentiment, 'score': main_score, 'confidence': main_confidence}] + reply_results
    positive = sum(1 for r in all_results if r['sentiment'] == 'Positive')
    negative = sum(1 for r in all_results if r['sentiment'] == 'Negative')
    if positive > negative:
        topic_sentiment = 'Positive'
    elif negative > positive:
        topic_sentiment = 'Negative'
    else:
        topic_sentiment = 'Neutral'

    scores = [main_score * 2] + [r['score'] for r in reply_results]
    weighted_score = sum(scores) / (2 + len(reply_results)) if reply_results else main_score
    max_urgency = max([main_urgency] + [calculate_urgency('', r['sentiment']) for r in reply_results])
    avg_confidence = sum(r.get('confidence', 0.0) for r in all_results) / len(all_results)

    return {
        'sentiment': topic_sentiment,
        'sentiment_score': round(weighted_score, 3),
        'urgency_score': max_urgency,
        'confidence': round(avg_confidence, 2),
        'review_required': False,
        'model_version': 'HTU-Sentiment-v4-unified',
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