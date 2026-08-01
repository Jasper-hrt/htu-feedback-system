"""
unknown_detector.py

Detects words that are not currently included
in the custom sentiment lexicon.
"""

import re

from sentiment.custom_lexicon import CustomLexiconManager


class UnknownWordDetector:

    def __init__(self):

        # Load the custom sentiment dictionary
        self.lexicon_manager = CustomLexiconManager()

        # Common words that should not be treated
        # as important unknown sentiment words
        self.common_words = {
            "the", "a", "an", "and", "or", "but",
            "is", "are", "was", "were",
            "i", "we", "you", "they",
            "this", "that", "these", "those",
            "to", "of", "in", "on", "at",
            "for", "with", "from", "by",
            "my", "our", "your", "their",
            "it", "be", "been", "being",
            "have", "has", "had",
            "do", "does", "did"
        }

    def detect(self, text: str) -> list:

        # Convert the text into lowercase words
        words = re.findall(
            r"\b[a-zA-Z]+\b",
            text.lower()
        )

        unknown_words = []

        for word in words:

            # Ignore short words
            if len(word) <= 2:
                continue

            # Ignore common grammar words
            if word in self.common_words:
                continue

            # Check whether the word is in
            # the custom sentiment dictionary
            if word not in self.lexicon_manager.lexicon:

                # Avoid adding the same word twice
                if word not in unknown_words:
                    unknown_words.append(word)

        return unknown_words