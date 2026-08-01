from textblob import TextBlob


class TextBlobEngine:

    def analyze(self, text):

        blob = TextBlob(text)

        return {
            "polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity
        }