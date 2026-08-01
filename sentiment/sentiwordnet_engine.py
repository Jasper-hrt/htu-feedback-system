from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.tokenize import word_tokenize


class SentiWordNetEngine:

    def penn_to_wn(self, tag):
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

        tokens = word_tokenize(text.lower())

        tagged = pos_tag(tokens)

        score = 0
        count = 0

        for word, tag in tagged:

            wn_tag = self.penn_to_wn(tag)

            if wn_tag is None:
                continue

            synsets = wordnet.synsets(word, pos=wn_tag)

            if not synsets:
                continue

            try:
                senti = swn.senti_synset(synsets[0].name())

                score += senti.pos_score() - senti.neg_score()

                count += 1

            except:
                continue

        if count == 0:
            return 0.0

        return score / count