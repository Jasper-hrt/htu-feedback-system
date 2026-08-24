"""
sentiwordnet_engine.py

Resilient SentiWordNet sentiment engine.

NLTK resources (wordnet, sentiwordnet, punkt, averaged_perceptron_tagger,
omw-1.4) are now imported *lazily* inside methods rather than at module load.
This prevents the entire app from crashing at startup when the corpora are
missing (e.g. on a fresh Render deploy before the NLTK download completes).

If the required NLTK data is unavailable, `analyze()` degrades gracefully to a
neutral score (0.0) instead of raising, so feedback submission never fails.
"""


class SentiWordNetEngine:

    _nltk_ok = None  # cached availability flag (None = not yet checked)

    def _load_nltk(self):
        """Lazily import NLTK and verify the required corpora are available.

        Returns (swn, wordnet, pos_tag, word_tokenize) or (None,)*4 if the
        resources are missing. The result is cached so we only probe once.
        """
        if SentiWordNetEngine._nltk_ok is not None:
            if not SentiWordNetEngine._nltk_ok:
                return (None, None, None, None)
            try:
                from nltk.corpus import sentiwordnet as swn
                from nltk.corpus import wordnet
                from nltk import pos_tag
                from nltk.tokenize import word_tokenize
                return (swn, wordnet, pos_tag, word_tokenize)
            except Exception:
                SentiWordNetEngine._nltk_ok = False
                return (None, None, None, None)

        # First probe: try to import + verify resources are downloadable/lookable.
        try:
            from nltk.corpus import sentiwordnet as swn
            from nltk.corpus import wordnet
            from nltk import pos_tag
            from nltk.tokenize import word_tokenize

            # Force a lookup so missing data raises here (cached as unavailable).
            _ = wordnet.synsets('test')
            _ = swn.senti_synset('good.a.01') if hasattr(swn, 'senti_synset') else None

            SentiWordNetEngine._nltk_ok = True
            return (swn, wordnet, pos_tag, word_tokenize)
        except Exception:
            SentiWordNetEngine._nltk_ok = False
            return (None, None, None, None)

    def penn_to_wn(self, tag, wordnet):
        if wordnet is None:
            return None
        if tag.startswith("J"):
            return wordnet.ADJ
        elif tag.startswith("V"):
            return wordnet.VERB
        elif tag.startswith("N"):
            return wordnet.NOUN
        elif tag.startswith("R"):
            return wordnet.ADV
        return None

    def analyze(self, text):
        swn, wordnet, pos_tag, word_tokenize = self._load_nltk()

        # Missing NLTK resources are an unavailable engine, not neutral evidence.
        if swn is None or wordnet is None or pos_tag is None or word_tokenize is None:
            return None

        try:
            tokens = word_tokenize(str(text).lower())
            tagged = pos_tag(tokens)
        except Exception:
            return None

        word_scores = []

        for word, tag in tagged:
            wn_tag = self.penn_to_wn(tag, wordnet)
            if wn_tag is None:
                continue

            try:
                # Prefer synsets whose POS matches the tagged token; fall back
                # to every sense if no POS-matched sense exists. Averaging over
                # senses (instead of trusting synsets[0]) smooths out wrong-sense
                # noise that previously made this engine the noisiest voter.
                synsets = wordnet.synsets(word, pos=wn_tag) or wordnet.synsets(word)
            except Exception:
                continue

            if not synsets:
                continue

            senses = []
            for syn in synsets:
                try:
                    senti = swn.senti_synset(syn.name())
                    senses.append(senti.pos_score() - senti.neg_score())
                except Exception:
                    continue

            if not senses:
                continue

            # Use the informative (non-zero) senses when present; they carry the
            # actual polarity. Fall back to the raw average otherwise.
            informative = [s for s in senses if s != 0.0]
            word_scores.append(
                sum(informative) / len(informative) if informative else sum(senses) / len(senses)
            )

        if not word_scores:
            return None

        return sum(word_scores) / len(word_scores)
