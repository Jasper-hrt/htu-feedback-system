"""Pluggable domain-model contract for HTU sentiment.

A future trained classifier can implement this interface and be inserted into
HybridSentimentEngine without changing the application routes or database.
The current release uses the context/rule layer plus the available statistical
engines, which keeps deployment lightweight and deterministic.
"""

from typing import Protocol, Dict, Any


class DomainSentimentModel(Protocol):
    model_version: str

    def predict(self, text: str) -> Dict[str, Any]:
        """Return label, score and confidence for normalized HTU feedback."""
        ...
