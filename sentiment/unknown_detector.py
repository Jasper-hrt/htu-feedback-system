"""
unknown_detector.py

Detects words that are not currently included in the custom sentiment
lexicon, and (via scan_feedback_for_gaps) aggregates that signal across
real submitted feedback so lexicon tuning can be driven by evidence
instead of guesswork.

Deliberately does NOT use nltk.corpus.stopwords -- app.py's NLTK
downloader is a best-effort background thread that isn't guaranteed to
be ready on a cold start (see _ensure_nltk_data), and this module
should work reliably without depending on that. The stopword list
below is self-contained instead.
"""

import re
from collections import Counter

from sentiment.custom_lexicon import CustomLexiconManager


# Common English function words (articles, pronouns, prepositions,
# conjunctions, auxiliary/modal verbs, etc.) that carry no sentiment on
# their own and would otherwise swamp the "unknown word" results with
# noise. Not meant to be exhaustive -- just enough to keep the gap
# report focused on words worth a human's attention.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "nor", "so", "yet",
    "is", "are", "was", "were", "be", "been", "being", "am",
    "i", "we", "you", "he", "she", "it", "they", "me", "him", "her",
    "us", "them", "my", "our", "your", "his", "its", "their", "mine",
    "ours", "yours", "theirs", "myself", "ourselves", "yourself",
    "yourselves", "himself", "herself", "itself", "themselves",
    "this", "that", "these", "those",
    "to", "of", "in", "on", "at", "for", "with", "from", "by", "about",
    "as", "into", "onto", "over", "under", "between", "through",
    "during", "before", "after", "above", "below", "up", "down",
    "out", "off", "again", "further", "once", "here", "there",
    "when", "where", "why", "how", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "only", "own",
    "same", "than", "too", "very", "just", "also",
    "have", "has", "had", "having",
    "do", "does", "did", "doing",
    "will", "would", "shall", "should", "can", "could", "may", "might",
    "must", "ought",
    "not", "no", "nor",
    "if", "because", "while", "although", "though", "since", "unless",
    "which", "who", "whom", "whose", "what",
    "get", "got", "getting", "one", "two", "like", "well", "really",
    "im", "dont", "didnt", "cant", "wont", "youre", "theyre", "ive",
    "student", "students", "htu", "campus", "please",
    "thanks", "thank", "hope", "want", "wanted", "need", "needed",
    "make", "made", "know", "think", "going", "used", "use", "even",
}


class UnknownWordDetector:

    def __init__(self):
        self.lexicon_manager = CustomLexiconManager()
        self.common_words = _STOPWORDS

    def detect(self, text: str) -> list:
        """Return the list of unique words in text that are neither a
        stopword nor present in the custom lexicon."""
        if not text:
            return []

        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())

        unknown_words = []
        for word in words:
            if len(word) <= 2:
                continue
            if word in self.common_words:
                continue
            if word not in self.lexicon_manager.lexicon:
                if word not in unknown_words:
                    unknown_words.append(word)

        return unknown_words

    def scan_feedback_for_gaps(self, feedback_items, top_n=40, snippet_len=120):
        """
        Aggregate unknown-word frequency across a batch of real feedback.

        feedback_items: iterable of (text, sentiment) tuples.

        Returns a dict:
          - 'scanned_count': how many feedback items were scanned
          - 'neutral_gap_words': [{'word', 'count', 'example'}, ...] --
                words found in feedback that came out Neutral, sorted by
                frequency. These are the strongest candidates: a Neutral
                result is often exactly what happens when nothing in any
                engine's vocabulary recognised the text, so a word that
                shows up often here is a real accuracy gap, not just a
                vocabulary gap.
          - 'all_gap_words': the same aggregation across every feedback
                item regardless of label, for broader lexicon coverage.

        Each unknown word is only counted once per feedback item, so one
        long message repeating a word doesn't dominate the ranking.
        """
        neutral_counts = Counter()
        all_counts = Counter()
        neutral_examples = {}
        all_examples = {}
        scanned = 0

        for text, sentiment in feedback_items:
            if not text:
                continue
            scanned += 1
            unknown_words = self.detect(text)
            snippet = text.strip()
            if len(snippet) > snippet_len:
                snippet = snippet[:snippet_len].rsplit(" ", 1)[0] + "…"

            for word in unknown_words:
                all_counts[word] += 1
                if word not in all_examples:
                    all_examples[word] = snippet
                if sentiment == "Neutral":
                    neutral_counts[word] += 1
                    if word not in neutral_examples:
                        neutral_examples[word] = snippet

        def _to_ranked_list(counts, examples):
            return [
                {"word": word, "count": count, "example": examples.get(word, "")}
                for word, count in counts.most_common(top_n)
            ]

        return {
            "scanned_count": scanned,
            "neutral_gap_words": _to_ranked_list(neutral_counts, neutral_examples),
            "all_gap_words": _to_ranked_list(all_counts, all_examples),
        }
