from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class VaderEngine:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze(self, text: str):
        scores = self.analyzer.polarity_scores(text)

        return {
            "positive": scores["pos"],
            "negative": scores["neg"],
            "neutral": scores["neu"],
            "compound": scores["compound"]
        }