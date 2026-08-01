"""
preprocessing.py

Handles text cleaning before sentiment analysis.

Enhanced with:
- Emoji-to-text conversion
- Basic spelling correction (lightweight, no external dependency)
- Negation handling (marks negated words for downstream)
- Lemmatization via NLTK WordNet
- Hashtag segmentation (#NoWaterInHostel -> no water in hostel)
- Character repetition normalization
- URL, email, mention stripping
"""

import re
from typing import List, Tuple


# ==================== EMOJI MAP ====================

EMOJI_MAP = {
    # Smileys & Positive
    "😀": "happy", "😃": "happy", "😄": "happy", "😁": "happy",
    "😊": "happy", "😇": "innocent", "🥰": "loving", "😍": "loving",
    "🤩": "impressed", "😌": "relieved", "😋": "joyful",
    "😎": "cool", "🤗": "hugging", "🙂": "smiling",
    "😉": "winking", "😛": "playful", "😜": "playful",

    # Positive gestures
    "👍": "good", "👎": "bad", "🙌": "celebrating",
    "👏": "applause", "💪": "strong", "❤️": "love",
    "💕": "love", "💖": "love", "💗": "love",
    "✨": "amazing", "⭐": "excellent", "🌟": "excellent",
    "🎉": "celebrating", "🎊": "celebrating", "🥳": "celebrating",
    "🔥": "awesome", "✅": "completed", "✔️": "completed",

    # Negative / Sad
    "😢": "sad", "😭": "crying", "😞": "disappointed",
    "😔": "sad", "😟": "worried", "😕": "confused",
    "🙁": "unhappy", "😣": "struggling", "😖": "frustrated",
    "😫": "tired", "🥺": "pleading",
    "😩": "exhausted", "😤": "frustrated",
    "😠": "angry", "😡": "angry", "🤬": "angry",
    "💔": "heartbroken", "💀": "dead",
    "😰": "anxious", "😨": "scared", "😱": "terrified",
    "😬": "uncomfortable", "😧": "pained",

    # Neutral / Misc
    "😐": "neutral", "😶": "silent", "🤔": "thinking",
    "🙄": "annoyed", "😒": "unimpressed", "😏": "smug",
    "😴": "sleeping", "🤒": "sick", "🤕": "injured",
    "🥴": "confused",

    # Symbols
    "💡": "idea", "📢": "announcement", "🔔": "notification",
    "⚠️": "warning", "🚨": "emergency", "🆘": "help",
    "🚩": "flag", "📌": "important",
    "🔴": "critical", "🟡": "warning", "🟢": "good",

    # Facilities / Places
    "🏠": "home", "🏢": "building", "🏫": "school",
    "🍔": "food", "🍕": "food", "🍽️": "food",
    "🚌": "bus", "🚗": "car", "🚪": "door",
    "🚻": "toilet", "🚾": "toilet",
    "📚": "study", "💻": "computer", "📱": "phone",
    "🔑": "key", "🔒": "locked",

    # Actions
    "📝": "feedback", "📋": "form", "📧": "email",
    "📞": "call", "📩": "message",
    "⏰": "time", "⌛": "waiting",

    # Nature / Weather
    "🌧️": "rain", "☀️": "sunny", "🌡️": "temperature",
}


# ==================== EMOJI PATTERN ====================

# Build regex pattern to match emojis
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F300-\U0001F5FF"  # Misc Symbols & Pictographs
    "\U0001F680-\U0001F6FF"  # Transport & Map
    "\U0001F1E0-\U0001F1FF"  # Flags
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251"  # Enclosed
    "\U0001F900-\U0001F9FF"  # Supplemental
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols Extended-A
    "\u200d"                  # Zero-width joiner
    "\ufe0f"                  # Variation selector
    "\u2600-\u26FF"           # Misc symbols
    "\u2700-\u27BF"           # Dingbats
    "]+",
    flags=re.UNICODE
)


def _replace_emoji(text: str) -> str:
    """Replace emojis with their text equivalents."""
    def _replacer(match):
        emoji_seq = match.group(0)
        # Try each emoji char in the sequence
        for ch in emoji_seq:
            if ch in EMOJI_MAP:
                return f" {EMOJI_MAP[ch]} "
        return " "  # Unknown emoji, replace with space
    return _EMOJI_PATTERN.sub(_replacer, text)


# ==================== SPELLING CORRECTION ====================

# Common misspellings in university feedback context
_COMMON_MISSPELLINGS = {
    # Facilities
    "wifi": "wifi",
    "wi-fi": "wifi",
    "accomodation": "accommodation",
    "accomodation": "accommodation",
    "acommodation": "accommodation",
    "dorm": "dormitory",
    "dormetry": "dormitory",
    "dormitary": "dormitory",
    "cafetaria": "cafeteria",
    "cafateria": "cafeteria",
    "librery": "library",
    "libary": "library",
    "libary": "library",
    "clasroom": "classroom",
    "classrom": "classroom",
    "laboratory": "laboratory",
    "labratory": "laboratory",
    "campus": "campus",
    "campuss": "campus",
    "lecture": "lecture",
    "lectur": "lecture",
    "lectural": "lecturer",
    "lec": "lecture",
    "leacture": "lecture",
    "leacturer": "lecturer",

    # Academic
    "assignment": "assignment",
    "assignmnt": "assignment",
    "assignmet": "assignment",
    "examination": "examination",
    "exams": "exams",
    "exam": "exam",
    "semester": "semester",
    "semster": "semester",
    "semestre": "semester",
    "curriculum": "curriculum",
    "curiculum": "curriculum",
    "syllabus": "syllabus",
    "sylabus": "syllabus",
    "timetable": "timetable",
    "time-table": "timetable",
    "time table": "timetable",
    "schdule": "schedule",
    "skedule": "schedule",
    "deadline": "deadline",
    "dateline": "deadline",
    "grading": "grading",
    "grader": "grading",

    # Sentiment words
    "frustrated": "frustrated",
    "frustrating": "frustrating",
    "frustation": "frustration",
    "disapointed": "disappointed",
    "disappointing": "disappointing",
    "disapointing": "disappointing",
    "embarrasing": "embarrassing",
    "embarassing": "embarrassing",
    "unbelieveable": "unbelievable",
    "unbeleivable": "unbelievable",
    "acheive": "achieve",
    "acheiving": "achieving",
    "acheivement": "achievement",

    # Technology
    "intenet": "internet",
    "interent": "internet",
    "intrnet": "internet",
    "connecton": "connection",
    "conection": "connection",
    "disconect": "disconnect",
    "disconected": "disconnected",
    "availble": "available",
    "availabe": "available",
    "avaliable": "available",
    "server": "server",
    "servver": "server",

    # Common errors
    "becuase": "because",
    "becuz": "because",
    "recieve": "receive",
    "recived": "received",
    "recieved": "received",
    "teh": "the",
    "adn": "and",
    "waht": "what",
    "thier": "their",
    "their": "their",
    "thay": "they",
    "wierd": "weird",
    "weird": "weird",
    "definately": "definitely",
    "definitly": "definitely",
    "definately": "definitely",
    "tommorow": "tomorrow",
    "tommorrow": "tomorrow",
    "calender": "calendar",
    "calandar": "calendar",
    "government": "government",
    "goverment": "government",
    "environmnt": "environment",
    "enviornment": "environment",
    "occuring": "occurring",
    "occured": "occurred",
    "ocurred": "occurred",
    "oportunity": "opportunity",
    "oppurtunity": "opportunity",
    "neccessary": "necessary",
    "necesary": "necessary",
    "again": "again",
    "agian": "again",
    "alot": "a lot",
}


def _correct_spelling(text: str) -> str:
    """Correct common misspellings using a lookup dictionary."""
    words = text.split()
    corrected = []
    for word in words:
        clean_word = word.strip(".,!?;:\"'()[]{}")
        punct_before = word[:len(word) - len(clean_word)]
        punct_after = word[len(clean_word):]
        lower_word = clean_word.lower()

        if lower_word in _COMMON_MISSPELLINGS:
            corrected_word = _COMMON_MISSPELLINGS[lower_word]
            # Preserve original capitalization pattern
            if clean_word.istitle():
                corrected_word = corrected_word.title()
            elif clean_word.isupper() and len(clean_word) > 1:
                corrected_word = corrected_word.upper()
            corrected.append(punct_before + corrected_word + punct_after)
        else:
            corrected.append(word)
    return " ".join(corrected)


# ==================== NEGATION HANDLING ====================

# Negation words and their contracted forms
_NEGATION_WORDS = {
    "not", "no", "never", "neither", "nor", "nobody",
    "nothing", "nowhere", "hardly", "barely", "scarcely",
    "n't", "don't", "doesn't", "didn't", "won't", "wouldn't",
    "shouldn't", "couldn't", "isn't", "aren't", "wasn't",
    "weren't", "haven't", "hasn't", "hadn't", "can't",
    "cannot", "mustn't", "needn't", "daren't",
}

# Words that stop negation scope
_NEGATION_STOPPERS = {
    "but", "however", "although", "though", "yet",
    "nevertheless", "nonetheless", "despite", ".",
    "!", "?", ";",
}


def _apply_negation_nrc(emotion_scores: dict, words: List[str]) -> dict:
    """
    Apply negation scope to emotion scores.
    Flips the dominant emotion if negation is detected.
    """
    negated = False
    negation_countdown = 0

    for word in words:
        clean_word = word.strip(".,!?;:\"'()[]{}").lower()

        if clean_word in _NEGATION_WORDS or clean_word.endswith("n't"):
            negated = True
            negation_countdown = 3  # Flip next 3 sentiment words
            continue

        if clean_word in _NEGATION_STOPPERS:
            negated = False
            negation_countdown = 0
            continue

        if negated and negation_countdown > 0:
            negation_countdown -= 1
            if negation_countdown == 0:
                negated = False

    return emotion_scores


# ==================== HASHTAG SEGMENTATION ====================

def _segment_hashtag(tag: str) -> str:
    """
    Segment a hashtag like #NoWaterInHostel into words.

    Uses:
    1. PascalCase splitting
    2. Underscore splitting
    3. Number boundary splitting
    """
    # Remove the #
    if tag.startswith("#"):
        tag = tag[1:]

    # Split on underscores
    if "_" in tag:
        parts = tag.split("_")
        return " ".join(p.lower() for p in parts if p)

    # Split on PascalCase/camelCase boundaries
    # "NoWaterInHostel" -> ["No", "Water", "In", "Hostel"]
    parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\b)|[0-9]+", tag)
    if len(parts) > 1:
        return " ".join(p.lower() for p in parts)

    # Split on number boundaries (NoWater2Hostel)
    parts = re.split(r"(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])", tag)
    parts = [re.sub(r"([a-z])([A-Z])", r"\1 \2", p).lower() for p in parts]
    parts = [p for p in parts if p]
    if len(parts) > 1:
        return " ".join(parts)

    # Single word hashtag - keep as is
    return tag.lower()


# ==================== CHARACTER REPETITION NORMALIZATION ====================

def _normalize_repetition(text: str) -> str:
    """Normalize repeated characters (sooooo -> soo, noooo -> noo)."""
    # For 3+ repeated characters, reduce to 2
    text = re.sub(r"(\w)\1{2,}", r"\1\1", text)
    return text


# ==================== MAIN PREPROCESSOR ====================

class TextPreprocessor:

    def __init__(self):
        self._lemmatizer = None  # Lazy-loaded

    def _get_lemmatizer(self):
        """Lazy-load WordNet lemmatizer to avoid startup slowdown."""
        if self._lemmatizer is None:
            try:
                from nltk.stem import WordNetLemmatizer
                import nltk
                try:
                    nltk.data.find('corpora/wordnet')
                except LookupError:
                    nltk.download('wordnet', quiet=True)
                self._lemmatizer = WordNetLemmatizer()
            except Exception:
                self._lemmatizer = None
        return self._lemmatizer

    def _lemmatize(self, text: str) -> str:
        """Apply lemmatization to normalize word forms.

        Uses NLTK's WordNetLemmatizer if available.
        Falls back gracefully if not installed.
        """
        lemmatizer = self._get_lemmatizer()
        if lemmatizer is None:
            return text

        words = text.split()
        lemmatized = []
        for word in words:
            clean_word = word.strip(".,!?;:\"'()[]{}")
            try:
                # Try verb first (most words in feedback are verbs/nouns)
                lemma = lemmatizer.lemmatize(clean_word, 'v')
                if lemma == clean_word:
                    lemma = lemmatizer.lemmatize(clean_word, 'n')
                if lemma == clean_word:
                    lemma = lemmatizer.lemmatize(clean_word, 'a')

                # Preserve original casing/punctuation
                if clean_word.istitle():
                    lemma = lemma.title()
                if clean_word.isupper() and len(clean_word) > 1:
                    lemma = lemma.upper()

                # Re-attach punctuation
                prefix = word[:len(word) - len(clean_word)]
                suffix = word[len(clean_word):]
                lemmatized.append(prefix + lemma + suffix)
            except Exception:
                lemmatized.append(word)
        return " ".join(lemmatized)

    def clean(self, text: str) -> str:
        """
        Clean the input text before analysis.

        Pipeline:
        1. Strip whitespace, handle None
        2. Replace emojis with text
        3. Segment hashtags
        4. Remove URLs and emails
        5. Remove @mentions (keep # content via segmentation)
        6. Expand slang/acronyms (via cleaning.py)
        7. Correct common misspellings
        8. Lowercase
        9. Remove special characters (keep letters, numbers, spaces)
        10. Normalize character repetition
        11. Lemmatize
        12. Remove extra whitespace
        13. Remove single-character tokens
        """
        if not text:
            return ""

        t = str(text).strip()

        # Step 1: Replace emojis with text equivalents
        t = _replace_emoji(t)

        # Step 2: Segment hashtags (#NoWaterInHostel -> no water in hostel)
        def _segment_hashtags(match):
            return _segment_hashtag(match.group(0))

        t = re.sub(r"#\w+", _segment_hashtags, t)

        # Step 3: Remove URLs
        t = re.sub(r"http\S+|www\S+|https\S+", " ", t, flags=re.IGNORECASE)

        # Step 4: Remove email addresses
        t = re.sub(r"\S+@\S+", " ", t)

        # Step 5: Remove @mentions
        t = re.sub(r"@\w+", " ", t)

        # Step 6: Expand slang/acronyms
        from cleaning import expand_slang
        t = expand_slang(t)

        # Step 7: Correct common misspellings
        t = _correct_spelling(t)

        # Step 8: Lowercase
        t = t.lower()

        # Step 9: Remove unwanted special characters
        # Keep letters, numbers, spaces, and basic punctuation
        t = re.sub(r"[^a-zA-Z0-9\s]", " ", t)

        # Step 10: Normalize character repetition
        t = _normalize_repetition(t)

        # Step 11: Lemmatize
        t = self._lemmatize(t)

        # Step 12: Normalize whitespace
        t = re.sub(r"\s+", " ", t).strip()

        # Step 13: Remove single-character tokens (except meaningful ones)
        meaningful_singles = {"a", "i"}
        t = " ".join(
            tok for tok in t.split()
            if len(tok) > 1 or tok in meaningful_singles
        )

        return t


# ==================== UTILITY FUNCTIONS ====================

def get_negation_tokens(text: str) -> List[str]:
    """Extract negation words from text to help downstream analysis."""
    words = text.lower().split()
    negations = []
    for word in words:
        clean_word = word.strip(".,!?;:\"'()[]{}")
        if clean_word in _NEGATION_WORDS or clean_word.endswith("n't"):
            negations.append(clean_word)
    return negations


def detect_emoji_count(text: str) -> dict:
    """Count emojis in text and categorize them."""
    positive_emojis = {"😀", "😃", "😄", "😁", "😊", "😍", "🥰", "❤️", "👍", "🎉", "🔥", "✨", "⭐", "🌟"}
    negative_emojis = {"😢", "😭", "😞", "😔", "😠", "😡", "🤬", "💔", "😨", "😰", "😱"}
    neutral_emojis = {"😐", "😶", "🤔", "🙄", "😴", "💡"}

    pos_count = 0
    neg_count = 0
    neu_count = 0

    for ch in text:
        if ch in positive_emojis:
            pos_count += 1
        elif ch in negative_emojis:
            neg_count += 1
        elif ch in neutral_emojis:
            neu_count += 1

    return {
        "positive": pos_count,
        "negative": neg_count,
        "neutral": neu_count,
        "total": pos_count + neg_count + neu_count
    }

