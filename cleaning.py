import re

# Reuse the same negation-contraction expansion used by the feedback
# preprocessing pipeline so "isn't stable" doesn't collapse into the bare,
# non-negated token "isn stable" once apostrophes are stripped below.
from sentiment.preprocessing import _expand_contractions


# Curated profanity list. Replace with NEUTRAL alternatives.
# Keep lowercase keys; matching is case-insensitive.
PROFANITY = {
    "fuck": "inappropriate language",
    "shit": "inappropriate language",
    "bitch": "insult",
    "asshole": "insult",
    "dick": "insult",
    "pussy": "insult",
    "cunt": "insult",
    "motherfucker": "inappropriate language",
    "whore": "insult",
    "slut": "insult",
    "fag": "insult",
    "faggot": "insult",
}


# Common obfuscation (leet/spacing) patterns.
OBFUSCATION_CHARS = {
    "a": r"[a@4]",
    "e": r"[e3]",
    "i": r"[i1!]",
    "o": r"[o0]",
    "u": r"[uυv]",
    "s": r"[s5$]",
    "t": r"[t7+]",
    "h": r"[h#]",
    "f": r"[fph]",
    "c": r"[c(k{2}|c)]+",  # permissive; not perfect
}


def expand_slang(text: str) -> str:
    """Expand common internet acronyms and slang (including Ghanaian/common usage).

    NOTE: Multi-word phrases are sorted before single words to prevent
    partial word replacement (e.g. "chop bar" before "chop").
    """
    if text is None:
        return ""

    t = str(text)

    # Build replacements list - multi-word phrases MUST come before single words
    replacements = [
        # ================================================================
        # Multi-word phrases (processed first - sorted by word count desc)
        # ================================================================

        # Ghanaian Pidgin multi-word
        (r"\bwo ho te sen\b", "how are you"),
        (r"\bmeda wo ase\b", "thank you"),
        (r"\bdey go\b", "going"),
        (r"\bdey come\b", "coming"),
        (r"\bdey work\b", "working"),
        (r"\bdey play\b", "playing"),
        (r"\bso tey\b", "until"),
        (r"\bno wahala\b", "no problem"),
        (r"\bwayo man\b", "fraudster"),
        (r"\bwo fa\b", "take"),
        (r"\bto bad\b", "too bad"),

        # Nigerian Pidgin multi-word
        (r"\bhow far now\b", "how are you"),
        (r"\bhow far\b", "how are you"),
        (r"\bwetin dey\b", "what's happening"),
        (r"\bno gree\b", "refuse"),
        (r"\bna wa o\b", "amazing"),
        (r"\bi no gree\b", "i refuse"),
        (r"\bi no sabi\b", "i don't know"),

        # Mood/attitude multi-word
        (r"\bgbas gig\b", "very bad"),
        (r"\bgbas man\b", "bad person"),
        (r"\bgbas woman\b", "bad woman"),

        # Campus-specific multi-word
        (r"\bchop bar\b", "restaurant"),
        (r"\bmain gate\b", "main entrance"),
        (r"\bfront desk\b", "reception"),
        (r"\bcommon room\b", "lounge"),
        (r"\bfood joint\b", "restaurant"),
        (r"\bfast food\b", "fast food"),

        # Common internet acronyms - multi-word
        (r"\blooking_forward\b", "looking forward"),
        (r"\bdon't_understand\b", "do not understand"),
        (r"\bno_idea\b", "no idea"),

        # ================================================================
        # Single-word replacements
        # ================================================================

        # pronouns/shortforms
        (r"\bu\b", "you"),
        (r"\bur\b", "your"),
        (r"\bw\b", "with"),
        (r"\br\b", "are"),
        (r"\bc\b", "see"),
        (r"\bb\b", "be"),

        # gratitude / engagement
        (r"\bpls\b", "please"),
        (r"\bplz\b", "please"),
        (r"\bthx\b", "thanks"),
        (r"\bty\b", "thank you"),
        (r"\btq\b", "thank you"),
        (r"\bthnks\b", "thanks"),
        (r"\btnx\b", "thanks"),

        # opinion/knowledge shorthand
        (r"\bimo\b", "in my opinion"),
        (r"\bimho\b", "in my honest opinion"),
        (r"\bidk\b", "i don't know"),
        (r"\bngl\b", "not gonna lie"),
        (r"\brn\b", "right now"),
        (r"\bkinda\b", "kind of"),
        (r"\bsorta\b", "sort of"),
        (r"\btbh\b", "to be honest"),
        (r"\btbf\b", "to be fair"),
        (r"\bwdym\b", "what do you mean"),
        (r"\bidc\b", "i don't care"),
        (r"\bwtf\b", "what the fuck"),
        (r"\bwth\b", "what the hell"),
        (r"\bafk\b", "away from keyboard"),
        (r"\birl\b", "in real life"),
        (r"\bfyi\b", "for your information"),
        (r"\bijbol\b", "just joking"),
        (r"\bjk\b", "just kidding"),
        (r"\bjs\b", "just saying"),

        # common acronyms
        (r"\bbrb\b", "be right back"),
        (r"\bomg\b", "oh my god"),
        (r"\bomfg\b", "oh my fucking god"),
        (r"\blol\b", "laughing out loud"),
        (r"\blmao\b", "laughing my ass off"),
        (r"\blmfao\b", "laughing my fucking ass off"),
        (r"\brofl\b", "rolling on the floor laughing"),
        (r"\bttyl\b", "talk to you later"),
        (r"\bafaik\b", "as far as i know"),
        (r"\basap\b", "as soon as possible"),
        (r"\bnp\b", "no problem"),
        (r"\bnvm\b", "never mind"),
        (r"\bgg\b", "good game"),
        (r"\bgl\b", "good luck"),
        (r"\bhf\b", "have fun"),
        (r"\bwb\b", "welcome back"),
        (r"\bwp\b", "well played"),
        (r"\bgtg\b", "got to go"),
        (r"\bg2g\b", "got to go"),
        (r"\bcya\b", "see you"),
        (r"\bc u\b", "see you"),
        (r"\bcu\b", "see you"),
        (r"\bgr8\b", "great"),
        (r"\b2day\b", "today"),
        (r"\b2moro\b", "tomorrow"),
        (r"\b2nite\b", "tonight"),
        (r"\bb4\b", "before"),
        (r"\bcuz\b", "because"),
        (r"\bcus\b", "because"),
        (r"\bcos\b", "because"),
        (r"\bcoz\b", "because"),

        # communication
        (r"\bdm\b", "direct message"),
        (r"\bpm\b", "private message"),
        (r"\bop\b", "original poster"),

        # ================================================================
        # Ghanaian & West African Slang
        # ================================================================
        (r"\bchale\b", "friend"),
        (r"\bchalle\b", "friend"),
        (r"\bchaley\b", "friend"),
        (r"\bmad\b", "very"),
        (r"\byawa\b", "problem"),
        (r"\byaa\b", "yes"),
        (r"\bnyansapo\b", "wisdom"),
        (r"\bsika\b", "money"),
        (r"\babro\b", "foreign"),
        (r"\bsakawa\b", "internet fraud"),
        (r"\btrotro\b", "public bus"),
        (r"\bobroni\b", "foreigner"),
        (r"\bakpeteshie\b", "local gin"),
        (r"\balata\b", "nigerian"),
        (r"\bkwasia\b", "fool"),
        (r"\bgbandi\b", "clumsy person"),

        # Ghanaian Pidgin / Common phrases - single words
        (r"\bdey\b", "is"),
        (r"\babeg\b", "my friend"),
        (r"\babi\b", "right"),
        (r"\bashawo\b", "petty trader"),
        (r"\bberekete\b", "hard"),
        (r"\bgb3\b", "good"),
        (r"\bgbege\b", "problem"),
        (r"\bhego\b", "wow"),
        (r"\bkpa\b", "kill"),
        (r"\bkrakra\b", "small"),
        (r"\bmaa\b", "small"),
        (r"\bna\b", "is"),
        (r"\bpadi\b", "friend"),
        (r"\bs3\b", "like"),
        (r"\bs3k3\b", "similar"),
        (r"\btee\b", "until"),
        (r"\bunu\b", "you people"),
        (r"\bwaa\b", "cry"),
        (r"\bwakye\b", "rice and beans"),
        (r"\bwayo\b", "trickery"),
        (r"\bye\b", "yes"),
        (r"\bɛyɛ\b", "it is good"),
        (r"\bɔkɔ\b", "let's go"),
        (r"\bɛte sen\b", "how is it"),
        (r"\bmedaase\b", "thank you"),

        # Nigerian Pidgin - single words
        (r"\bwetin\b", "what"),
        (r"\bwahala\b", "trouble"),
        (r"\bcomot\b", "leave"),
        (r"\baboki\b", "foolish"),
        (r"\basha\b", "sorry"),
        (r"\bjare\b", "please"),
        (r"\bsabi\b", "know"),
        (r"\bchop\b", "eat"),
        (r"\bdash\b", "give"),
        (r"\bokada\b", "okay"),
        (r"\botondo\b", "penis"),
        (r"\balaye\b", "meaning"),

        # Mood/attitude - single words
        (r"\bgbas\b", "garbage/trash"),

        # Campus-specific single words
        (r"\bhalls\b", "hall of residence"),
        (r"\bblock\b", "dormitory block"),
        (r"\bchapel\b", "church building"),
        (r"\bmosque\b", "prayer building"),
        (r"\bsrcla\b", "src local association"),
        (r"\bdlas\b", "departmental association"),
        (r"\bventilator\b", "fan"),
        (r"\bpop\b", "electricity outage"),
        (r"\bspot\b", "hangout place"),
        (r"\bbuka\b", "local restaurant"),
    ]

    for pattern, repl in replacements:
        t = re.sub(pattern, repl, t, flags=re.IGNORECASE)

    # Lightweight leetspeak expansions for digits.
    t = re.sub(r"\b1\b", "i", t)
    t = re.sub(r"\b0\b", "o", t)
    t = re.sub(r"\b3\b", "e", t)
    t = re.sub(r"\b4\b", "a", t)
    t = re.sub(r"\b5\b", "s", t)
    t = re.sub(r"\b2\b", "to", t)
    t = re.sub(r"\b8\b", "ate", t)

    return t


def _build_obfuscated_regex(token: str) -> re.Pattern:
    """Build a regex that matches obfuscated variants of a profanity token."""
    letters = list(token)

    out = []
    for ch in letters:
        ch_lower = ch.lower()
        if ch_lower in OBFUSCATION_CHARS:
            out.append(OBFUSCATION_CHARS[ch_lower])
        else:
            out.append(re.escape(ch_lower))

    sep = r"(?:[\W_]*?)"
    pattern = sep.join(out)
    return re.compile(rf"(?i)(?<!\w){pattern}(?!\w)")


# Profanity regexes are static, so compile them once and reuse. Building them
# on every clean_text() call (once per token per submission/chat message) was a
# measurable, avoidable cost on the hot path.
_PROFANITY_REGEXES = None


def _get_profanity_regexes() -> dict:
    global _PROFANITY_REGEXES
    if _PROFANITY_REGEXES is None:
        _PROFANITY_REGEXES = {tok: _build_obfuscated_regex(tok) for tok in PROFANITY}
    return _PROFANITY_REGEXES


# Precompile detection regexes.
# NOTE: previously this could be expensive/trigger pathological regex compilation
# for leetspeak/obfuscation. Detection regexes are not required for the current
# cleaning pipeline (we apply profanity replacement directly in clean_text),
# so we avoid compiling them at import time.
_DETECT_REGEXES = []


def clean_text(text: str) -> str:
    """Clean user text before sentiment/category analysis.

    Requirements implemented:
    - remove special characters/symbols
    - remove hashtag symbol but keep the word (#word -> word)
    - remove @mentions
    - expand internet acronyms + slang (including Ghanaian starter set)
    - replace profanity with neutral alternatives
    - remove extra whitespace, single characters, and repeated characters
    """
    if text is None:
        return ""

    t = str(text)

    # remove @mentions
    t = re.sub(r"@\w+", " ", t)

    # remove # symbol but keep the word
    t = re.sub(r"#([\w\u00C0-\u024F]+)", r"\1", t)

    # expand acronyms/slang
    t = expand_slang(t)

    # lowercase
    t = t.lower()

    # expand negation contractions (isn't -> is not) BEFORE punctuation is
    # stripped below, so the negation word survives instead of degrading
    # into a meaningless orphan token ("isn t" -> single-char "t" removed).
    t = _expand_contractions(t)

    # replace profanity with neutral alternatives
    for token, neutral in sorted(PROFANITY.items(), key=lambda kv: len(kv[0]), reverse=True):
        rx = _get_profanity_regexes()[token]
        t = rx.sub(neutral, t)

    # remove special characters/symbols (keep letters/numbers and whitespace)
    t = re.sub(r"[^0-9A-Za-z\u00C0-\u024F\s]", " ", t)

    # de-repeat elongated characters inside tokens (sooooo -> soo)
    def _dedupe_token(tok: str) -> str:
        return re.sub(r"(\w)\1{2,}", r"\1\1", tok)

    t = " ".join(_dedupe_token(tok) for tok in t.split())

    # normalize whitespace
    t = re.sub(r"\s+", " ", t).strip()

    # remove single-character tokens
    t = " ".join(tok for tok in t.split() if len(tok) > 1)

    return t


def censor_text(text: str, mask_char: str = "*") -> str:
    """Backward-compatible profanity censor.

    Note: clean_text() already replaces profanity with neutral alternatives.
    This function keeps the original signature used elsewhere.
    """
    if text is None:
        return ""

    # Replace with neutral alternatives as well.
    t = str(text)
    for token, neutral in sorted(PROFANITY.items(), key=lambda kv: len(kv[0]), reverse=True):
        rx = _get_profanity_regexes()[token]
        t = rx.sub(neutral, t)
    return t.strip()

