"""
download_nltk_data.py

Downloads all NLTK corpora and models required by the HTU SRC Feedback System.

Run this during the Render build step (before the app starts) to ensure
NLTK data is available in the environment.

Required by:
- sentiment/sentiwordnet_engine.py: wordnet, sentiwordnet, punkt, averaged_perceptron_tagger
- sentiment/preprocessing.py: wordnet (for lemmatization)
"""

import nltk
import sys

NLTK_DATA = [
    'wordnet',
    'punkt',
    'averaged_perceptron_tagger',
    'omw-1.4',  # Open Multilingual WordNet (required for sentiwordnet)
]

def download_all():
    print("=== Downloading NLTK data ===")
    for resource in NLTK_DATA:
        print(f"  Downloading {resource}...", end=' ')
        sys.stdout.flush()
        try:
            nltk.download(resource, quiet=False)
            print("✓")
        except Exception as e:
            print(f"✗ FAILED: {e}")
            # Don't exit - some downloads may succeed even if others fail
    print("=== NLTK download complete ===")

if __name__ == '__main__':
    download_all()
