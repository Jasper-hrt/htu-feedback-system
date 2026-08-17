"""Optional VADER adapter.

Returns None when VADER is unavailable instead of pretending that the text was
neutral.
"""

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception:
    SentimentIntensityAnalyzer = None


class VaderEngine:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer else None

    @property
    def available(self):
        return self.analyzer is not None

    def analyze(self, text: str):
        if self.analyzer is None:
            return None
        scores = self.analyzer.polarity_scores(text)
        return {
            "positive": scores["pos"],
            "negative": scores["neg"],
            "neutral": scores["neu"],
            "compound": scores["compound"]
        }
