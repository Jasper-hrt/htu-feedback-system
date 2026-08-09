from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from cleaning import clean_text as unified_clean_text, censor_text as unified_censor_text

# Authoritative critical-safety vocabulary shared across the whole stack.
# Guarantees kidnapping/assault/shooting/etc. are never downgraded to Neutral.
from sentiment.safety_vocabulary import (
    has_critical_safety,
    has_safety_concern,
    get_critical_terms,
    get_safety_concern_terms,
)

analyzer = SentimentIntensityAnalyzer()
from sentiment.hybrid_engine import HybridSentimentEngine

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
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        return 'Positive', compound
    elif compound <= -0.05:
        return 'Negative', compound
    else:
        return 'Neutral', compound


def get_sentiment_explanation(text, top_n=6):

    """Explain sentiment by listing tokens/phrases that likely influenced VADER.

    Implementation approach:
    - Use the cleaned text.
    - VADER contains an internal lexicon with word->valence.
    - We take lexicon entries that appear in the text, rank by absolute valence,
      and return them as the explanation.

    Note: This is not a perfect per-token SHAP-style explanation, but it provides
    a practical "why" box as requested.
    """
    if not text:
        return {
            'sentiment_explanation': [],
            'compound': 0.0
        }

    scores = analyzer.polarity_scores(text)
    compound = scores.get('compound', 0.0)

    # analyzer.lexicon is a dict of word -> valence score (float)
    lex = getattr(analyzer, 'lexicon', {})
    if not isinstance(lex, dict) or not lex:
        return {'sentiment_explanation': [], 'compound': compound}

    text_l = str(text).lower()

    candidates = []
    for word, val in lex.items():
        # Skip non-alphabetic tokens (mostly punctuation/rare) to reduce noise
        if not word or not isinstance(word, str):
            continue
        if word.strip() == '':
            continue
        # Avoid lexicon artifacts like single letters that can dominate by containment.
        if len(word.strip()) < 3:
            continue


        # Simple containment check; since text is already cleaned/lowercased,
        # this stays reasonably fast.
        if word in text_l:
            try:
                val_f = float(val)
            except Exception:
                continue
            candidates.append((word, val_f))

    # Rank by magnitude (how strong the word is), then favor words whose polarity matches decision.
    sentiment, _ = analyze_sentiment(text)
    sign = 1 if sentiment == 'Positive' else (-1 if sentiment == 'Negative' else 0)

    def sort_key(item):
        w, v = item
        match = 0
        if sign != 0:
            match = 1 if (v * sign) > 0 else 0
        return (match, abs(v),)

    candidates.sort(key=sort_key, reverse=True)

    influencers = []
    for w, v in candidates[:top_n]:
        influencers.append({
            'word': w,
            'impact': round(v, 3)
        })

    return {
        'sentiment_explanation': influencers,
        'compound': compound
    }


def calculate_urgency(text, sentiment):
    text_lower = text.lower()
    urgency = 1

    # A genuine critical safety incident (kidnapping, assault, shooting,
    # weapon, threat of violence, etc.) is ALWAYS an emergency.
    if has_critical_safety(text):
        return 5

    urgency_keywords = {
        5: ['emergency', 'danger', 'injured', 'unsafe', 'hazard', 'assault', 'harassment', 'urgent', 'critical',
            'gunshots', 'gunshot', 'shooting', 'weapon', 'violence', 'hostage', 'threat', 'armed', 'attack',
            'blood', 'injury', 'panic', 'fight'],
        4: ['weeks', 'ignored', 'still', 'not working', 'broken', 'no water', 'no power', 'flood', 'collapsed', 'fire'],
        3: ['delay', 'late', 'rude', 'unhelpful', 'expensive', 'overcharged', 'frustrated', 'annoying'],
        2: ['slow', 'small', 'noisy', 'crowded', 'uncomfortable'],
        1: ['suggestion', 'maybe', 'could improve', 'consider']
    }

    for score, keywords in urgency_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                urgency = max(urgency, score)

    if sentiment == 'Negative' and urgency < 3:
        urgency = 3

    return min(urgency, 5)


def get_urgency_explanation(text, sentiment):
    """Explain urgency by returning detected urgency keywords + how final score was derived.

    The UI expects:
      - Key Phrases Box: keywords detected from urgency lexicon
      - Urgency Explanation Box: why the final score (1-5) was chosen

    Mirrors calculate_urgency() logic so explanations stay consistent with scoring.
    """
    if text is None:
        text = ''

    text_lower = str(text).lower()

    # Critical safety incidents are always urgency 5 and should be reported
    # transparently in the explanation.
    if has_critical_safety(str(text)):
        critical_terms = [t for t in get_critical_terms() if t in text_lower]
        return {
            'urgency_keywords_detected': critical_terms[:12],
            'sentiment_boost_applied': False,
            'base_urgency_from_keywords': 5,
            'final_urgency': 5,
            'sentiment': sentiment,
            'critical_safety_override': True,
        }

    urgency_keywords = {
        5: ['emergency', 'danger', 'injured', 'unsafe', 'hazard', 'assault', 'harassment', 'urgent', 'critical',
            'gunshots', 'gunshot', 'shooting', 'weapon', 'violence', 'hostage', 'threat', 'armed', 'attack',
            'blood', 'injury', 'panic', 'fight'],
        4: ['weeks', 'ignored', 'still', 'not working', 'broken', 'no water', 'no power', 'flood', 'collapsed', 'fire'],
        3: ['delay', 'late', 'rude', 'unhelpful', 'expensive', 'overcharged', 'frustrated', 'annoying'],
        2: ['slow', 'small', 'noisy', 'crowded', 'uncomfortable'],
        1: ['suggestion', 'maybe', 'could improve', 'consider']
    }

    detected = []
    base_urgency = 1

    for score, keywords in urgency_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                detected.append(keyword)
                base_urgency = max(base_urgency, score)

    sentiment_boost_applied = False
    final_urgency = base_urgency

    if sentiment == 'Negative' and final_urgency < 3:
        sentiment_boost_applied = True
        final_urgency = 3

    final_urgency = min(final_urgency, 5)

    # De-duplicate while preserving order
    seen = set()
    detected_unique = []
    for k in detected:
        if k not in seen:
            seen.add(k)
            detected_unique.append(k)

    return {
        'urgency_keywords_detected': detected_unique[:12],
        'sentiment_boost_applied': sentiment_boost_applied,
        'base_urgency_from_keywords': base_urgency,
        'final_urgency': final_urgency,
        'sentiment': sentiment
    }


def detect_category(text):
    text_lower = text.lower()

    # A genuine critical safety incident always maps to the Safety category,
    # regardless of any incidental keywords from other categories.
    if has_critical_safety(str(text)):
        return 'Safety'

    category_keywords = {
        'Accommodation': ['hostel', 'dorm', 'room', 'bed', 'water', 'toilet', 'shower', 'accommodation', 'hall', 'bathroom', 'flood', 'leak'],
        'ICT/Wi-Fi': ['wifi', 'internet', 'network', 'connection', 'computer', 'lab', 'portal', 'slow', 'disconnect', 'server'],
        'Academics': ['lecturer', 'class', 'exam', 'course', 'assignment', 'timetable', 'curriculum', 'grade', 'result', 'lecture'],
        'Catering': ['food', 'canteen', 'cafeteria', 'meal', 'dining', 'hungry', 'price', 'restaurant', 'kitchen', 'cook'],
        'Facilities': ['library', 'classroom', 'building', 'elevator', 'light', 'fan', 'ac', 'chair', 'desk', 'projector'],
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
            if keyword in text_lower:
                scores[category] += 1
    
    if max(scores.values()) > 0:
        return max(scores, key=scores.get)
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
    cleaned = clean_text(message)
    sentiment, score = analyze_sentiment(cleaned)
    urgency = calculate_urgency(cleaned, sentiment)

    # Critical safety content must never be flagged Neutral/Positive in chat.
    if has_critical_safety(cleaned) and sentiment != 'Negative':
        sentiment = 'Negative'
        score = min(score, -0.8)

    is_flagged = (sentiment == 'Negative' and urgency >= 3)
    
    return {
        'sentiment': sentiment,
        'sentiment_score': round(score, 3),
        'urgency_score': urgency,
        'cleaned_message': cleaned,
        'is_flagged': is_flagged
    }

def analyze_topic(content, replies=[]):
    cleaned_main = clean_text(content)
    main_sentiment, main_score = analyze_sentiment(cleaned_main)
    main_urgency = calculate_urgency(cleaned_main, main_sentiment)

    # Critical safety content in the main post must never be Neutral/Positive.
    if has_critical_safety(cleaned_main) and main_sentiment != 'Negative':
        main_sentiment = 'Negative'
        main_score = min(main_score, -0.8)

    reply_sentiments = []
    reply_urgencies = []
    reply_scores = []
    for reply in replies:
        if hasattr(reply, 'content'):
            cleaned_reply = clean_text(reply.content)
            reply_sent, reply_score = analyze_sentiment(cleaned_reply)
            reply_urg = calculate_urgency(cleaned_reply, reply_sent)
            # Critical safety content in a reply is also forced Negative.
            if has_critical_safety(cleaned_reply) and reply_sent != 'Negative':
                reply_sent = 'Negative'
                reply_score = min(reply_score, -0.8)
            reply_sentiments.append(reply_sent)
            reply_urgencies.append(reply_urg)
            reply_scores.append(reply_score)
    
    all_sentiments = [main_sentiment] + reply_sentiments
    positive = sum(1 for s in all_sentiments if s == 'Positive')
    negative = sum(1 for s in all_sentiments if s == 'Negative')
    
    if positive > negative:
        topic_sentiment = 'Positive'
    elif negative > positive:
        topic_sentiment = 'Negative'
    else:
        topic_sentiment = 'Neutral'
    
    # Include reply sentiment scores (main post weighted higher).
    # We treat each reply as equally important, but the topic original content has 2x weight.
    weighted_score = (
        (main_score * 2) + sum(reply_scores)
    ) / (2 + len(reply_scores)) if reply_scores else main_score

    max_urgency = max([main_urgency] + reply_urgencies) if reply_urgencies else main_urgency
    
    return {
        'sentiment': topic_sentiment,
        'sentiment_score': round(weighted_score, 3),
        'urgency_score': max_urgency
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