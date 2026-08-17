"""Optional AFINN adapter."""

try:
    from afinn import Afinn
except Exception:
    Afinn = None


class AfinnEngine:
    def __init__(self):
        self.afinn = Afinn() if Afinn else None

    @property
    def available(self):
        return self.afinn is not None

    def analyze(self, text):
        if self.afinn is None:
            return None
        score = self.afinn.score(text)
        return max(-1, min(1, score / 5))
