"""
Download required NLTK data packages.
Run this once before starting the app: python download_nltk_data.py
"""
import nltk

def download_nltk_data():
    """Download all required NLTK data packages."""
    packages = [
        'punkt',
        'punkt_tab',
        'averaged_perceptron_tagger',
        'averaged_perceptron_tagger_eng',
        'wordnet',
        'sentiwordnet',
        'omw-1.4',
        'stopwords',
        'vader_lexicon',
    ]
    
    for package in packages:
        try:
            nltk.download(package, quiet=True)
            print(f"✓ {package}")
        except Exception as e:
            print(f"✗ {package}: {e}")
    
    print("\nNLTK data download complete!")

if __name__ == '__main__':
    download_nltk_data()
