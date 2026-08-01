from afinn import Afinn


class AfinnEngine:

    def __init__(self):
        self.afinn = Afinn()

    def analyze(self, text):

        score = self.afinn.score(text)

        # Normalize to approximately -1 to +1
        score = max(-1, min(1, score / 5))

        return score