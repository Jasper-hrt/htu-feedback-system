"""Optional TextBlob adapter."""

try:
    from textblob import TextBlob
except Exception:
    TextBlob = None


class TextBlobEngine:
    @property
    def available(self):
        return TextBlob is not None

    def analyze(self, text):
        if TextBlob is None:
            return None
        blob = TextBlob(text)
        return {
            "polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity
        }
