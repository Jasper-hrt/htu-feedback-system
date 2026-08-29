"""
custom_lexicon.py

Stores additional sentiment words and calculates
their sentiment score.

Expanded with 200+ domain-specific words for:
- HTU academic life, facilities, ICT, catering, hostel
- Ghanaian university slang
- Safety / security
- Student wellbeing / mental health
- Intensifiers and modifiers
"""

import re

from sentiment.safety_vocabulary import (
    CRITICAL_SAFETY_TERMS,
    SAFETY_CONCERN_TERMS,
    is_discussion_context,
)

# Lexicon entries below that duplicate a safety_vocabulary.py term -- these
# are the ones skipped when the text is discussing/learning about a safety
# topic rather than reporting an incident (see calculate_score docstring).
_SAFETY_VOCAB_OVERLAP = frozenset(CRITICAL_SAFETY_TERMS) | frozenset(SAFETY_CONCERN_TERMS)


class CustomLexiconManager:

    def __init__(self):

        # Custom sentiment dictionary
        # Scores range from -1.0 (most negative) to +1.0 (most positive)
        self.lexicon = {

            # ================================================================
            # Positive words - GENERAL (EXPANDED)
            # ================================================================
            "excellent": 0.9,
            "amazing": 0.9,
            "wonderful": 0.8,
            "fantastic": 0.8,
            "helpful": 0.6,
            "smooth": 0.5,
            "satisfied": 0.7,
            "efficient": 0.6,
            "brilliant": 0.85,
            "outstanding": 0.9,
            "superb": 0.85,
            "awesome": 0.8,
            "perfect": 0.85,
            "lovely": 0.7,
            "nice": 0.5,
            "good": 0.5,
            "great": 0.6,
            "better": 0.5,
            "best": 0.7,
            "improved": 0.6,
            "impressive": 0.75,
            "recommend": 0.6,
            "appreciate": 0.7,
            "grateful": 0.75,
            "thankful": 0.7,
            "pleased": 0.65,
            "delighted": 0.8,
            "happy": 0.7,
            "enjoy": 0.6,
            "love": 0.75,
            "like": 0.4,
            "prefer": 0.3,
            "comfortable": 0.5,
            "convenient": 0.5,
            "accessible": 0.4,
            "affordable": 0.5,
            "clean": 0.5,
            "tidy": 0.4,
            "organized": 0.5,
            "professional": 0.6,
            "supportive": 0.65,
            "responsive": 0.6,
            "timely": 0.5,
            "prompt": 0.5,
            "consistent": 0.4,
            "stable": 0.4,
            "reliable": 0.55,
            "effective": 0.55,
            "exceptional": 0.85,
            "magnificent": 0.9,
            "marvelous": 0.85,
            "splendid": 0.8,
            "terrific": 0.8,
            "fabulous": 0.85,
            "incredible": 0.8,
            "remarkable": 0.75,
            "phenomenal": 0.85,
            "stellar": 0.8,
            "superior": 0.7,
            "top_notch": 0.8,
            "world_class": 0.85,
            "first_rate": 0.8,
            "high_quality": 0.7,
            "well_done": 0.7,
            "well_made": 0.6,
            "well_run": 0.6,
            "user_friendly": 0.6,
            "seamless": 0.65,
            "flawless": 0.85,
            "faultless": 0.8,
            "ideal": 0.7,
            "exemplary": 0.8,
            "commendable": 0.7,
            "praiseworthy": 0.7,
            "worthwhile": 0.6,
            "valuable": 0.6,
            "useful": 0.5,
            "beneficial": 0.6,
            "favorable": 0.55,
            "positive": 0.5,
            "successful": 0.65,
            "productive": 0.55,
            "progressive": 0.5,
            "innovative": 0.6,
            "creative": 0.5,
            "inspiring": 0.7,
            "uplifting": 0.65,
            "refreshing": 0.55,
            "delightful": 0.7,
            "pleasant": 0.55,
            "enjoyable": 0.6,
            "satisfying": 0.65,
            "rewarding": 0.6,
            "fulfilling": 0.6,
            "gratifying": 0.65,
            "heartwarming": 0.7,
            "cheerful": 0.6,
            "joyful": 0.7,
            "blissful": 0.75,
            "ecstatic": 0.85,
            "elated": 0.8,
            "euphoric": 0.85,
            "overjoyed": 0.8,
            "thrilled": 0.75,
            "excited": 0.65,
            "enthusiastic": 0.6,
            "passionate": 0.55,
            "zealous": 0.5,
            "eager": 0.45,
            "optimistic": 0.55,
            "hopeful": 0.5,
            "confident": 0.5,
            "proud": 0.55,
            "accomplished": 0.6,
            "triumphant": 0.7,
            "victorious": 0.7,
            "empowering": 0.6,
            "liberating": 0.55,
            "enlightening": 0.6,
            "insightful": 0.55,
            "enriching": 0.6,
            "educational": 0.5,
            "informative": 0.5,
            "illuminating": 0.55,
            "revealing": 0.4,
            "eye_opening": 0.5,
            "thought_provoking": 0.45,
            "mind_blowing": 0.7,
            "breathtaking": 0.75,
            "awe_inspiring": 0.7,
            "spectacular": 0.75,
            "stunning": 0.7,
            "gorgeous": 0.65,
            "beautiful": 0.6,
            "elegant": 0.55,
            "graceful": 0.55,
            "charming": 0.55,
            "attractive": 0.5,
            "appealing": 0.5,
            "engaging": 0.55,
            "captivating": 0.6,
            "fascinating": 0.6,
            "intriguing": 0.5,
            "compelling": 0.55,
            "riveting": 0.6,
            "gripping": 0.55,
            "absorbing": 0.5,
            "enthralling": 0.65,
            "mesmerizing": 0.65,
            "spellbinding": 0.65,
            "magical": 0.6,
            "enchanting": 0.6,
            "wonderful": 0.75,
            "lovely": 0.65,
            "adorable": 0.6,
            "endearing": 0.55,
            "charming": 0.55,
            "delightful": 0.65,
            "sweet": 0.45,
            "kind": 0.5,
            "generous": 0.55,
            "caring": 0.55,
            "compassionate": 0.6,
            "empathetic": 0.55,
            "understanding": 0.5,
            "patient": 0.4,
            "thoughtful": 0.5,
            "considerate": 0.5,
            "respectful": 0.5,
            "courteous": 0.5,
            "polite": 0.45,
            "gracious": 0.55,
            "humble": 0.4,
            "modest": 0.35,
            "sincere": 0.5,
            "genuine": 0.5,
            "authentic": 0.5,
            "honest": 0.5,
            "trustworthy": 0.55,
            "dependable": 0.5,
            "loyal": 0.5,
            "faithful": 0.5,
            "devoted": 0.5,
            "committed": 0.45,
            "dedicated": 0.5,
            "hardworking": 0.5,
            "diligent": 0.5,
            "industrious": 0.5,
            "conscientious": 0.5,
            "meticulous": 0.5,
            "thorough": 0.45,
            "careful": 0.4,
            "cautious": 0.3,
            "prudent": 0.4,
            "wise": 0.5,
            "sensible": 0.45,
            "reasonable": 0.4,
            "fair": 0.4,
            "just": 0.45,
            "equitable": 0.45,
            "impartial": 0.4,
            "unbiased": 0.4,
            "objective": 0.35,
            "balanced": 0.4,
            "moderate": 0.3,
            "measured": 0.3,
            "restrained": 0.25,
            "tolerant": 0.4,
            "accepting": 0.4,
            "inclusive": 0.45,
            "welcoming": 0.5,
            "friendly": 0.5,
            "approachable": 0.45,
            "accessible": 0.4,
            "available": 0.35,
            "responsive": 0.5,
            "attentive": 0.5,
            "observant": 0.4,
            "perceptive": 0.45,
            "discerning": 0.4,
            "judicious": 0.45,
            "astute": 0.5,
            "shrewd": 0.4,
            "clever": 0.45,
            "ingenious": 0.55,
            "resourceful": 0.5,
            "innovative": 0.55,
            "inventive": 0.5,
            "original": 0.45,
            "unique": 0.4,
            "novel": 0.4,
            "fresh": 0.4,
            "new": 0.25,
            "modern": 0.4,
            "contemporary": 0.35,
            "current": 0.25,
            "up_to_date": 0.4,
            "advanced": 0.45,
            "sophisticated": 0.45,
            "refined": 0.4,
            "polished": 0.4,
            "sleek": 0.35,
            "stylish": 0.4,
            "trendy": 0.35,
            "fashionable": 0.35,
            "elegant": 0.5,
            "classy": 0.45,
            "posh": 0.4,
            "luxurious": 0.5,
            "premium": 0.45,
            "exclusive": 0.4,
            "elite": 0.45,
            "prestigious": 0.5,
            "renowned": 0.5,
            "celebrated": 0.55,
            "famous": 0.4,
            "notable": 0.4,
            "distinguished": 0.5,
            "eminent": 0.5,
            "prominent": 0.45,
            "leading": 0.45,
            "foremost": 0.5,
            "premier": 0.5,
            "prime": 0.45,
            "supreme": 0.55,
            "ultimate": 0.5,
            "paramount": 0.5,
            "preeminent": 0.55,
            "outstanding": 0.7,
            "exceptional": 0.75,
            "extraordinary": 0.7,
            "remarkable": 0.65,
            "notable": 0.5,
            "noteworthy": 0.5,
            "significant": 0.45,
            "important": 0.4,
            "meaningful": 0.45,
            "substantial": 0.4,
            "considerable": 0.35,
            "appreciable": 0.4,
            "noticeable": 0.35,
            "visible": 0.3,
            "evident": 0.3,
            "obvious": 0.25,
            "clear": 0.3,
            "plain": 0.25,
            "apparent": 0.3,
            "manifest": 0.35,
            "patent": 0.3,
            "unmistakable": 0.4,
            "undeniable": 0.4,
            "indisputable": 0.45,
            "irrefutable": 0.45,
            "incontrovertible": 0.45,
            "certain": 0.3,
            "sure": 0.3,
            "positive": 0.4,
            "definite": 0.35,
            "absolute": 0.4,
            "complete": 0.4,
            "total": 0.35,
            "full": 0.3,
            "entire": 0.3,
            "whole": 0.3,
            "comprehensive": 0.45,
            "extensive": 0.4,
            "broad": 0.3,
            "wide": 0.3,
            "vast": 0.35,
            "immense": 0.4,
            "enormous": 0.4,
            "huge": 0.35,
            "massive": 0.35,
            "substantial": 0.4,
            "considerable": 0.35,
            "significant": 0.4,
            "major": 0.35,
            "key": 0.3,
            "central": 0.3,
            "core": 0.3,
            "fundamental": 0.35,
            "essential": 0.4,
            "vital": 0.4,
            "critical": 0.35,
            "crucial": 0.4,
            "indispensable": 0.45,
            "necessary": 0.3,
            "needed": 0.3,
            "required": 0.3,
            "requisite": 0.3,
            "mandatory": 0.25,
            "obligatory": 0.25,
            "compulsory": 0.25,
            "imperative": 0.35,
            "urgent": 0.3,
            "pressing": 0.25,
            "exigent": 0.3,

            # ================================================================
            # Positive words - ACADEMIC
            # ================================================================
            "knowledgeable": 0.7,
            "engaging": 0.65,
            "informative": 0.6,
            "inspiring": 0.75,
            "motivating": 0.7,
            "enlightening": 0.7,
            "educational": 0.55,
            "insightful": 0.65,
            "thorough": 0.5,
            "comprehensive": 0.5,
            "well_structured": 0.6,
            "well_organized": 0.6,
            "fairness": 0.5,
            "fair": 0.4,
            "transparent": 0.5,
            "clear": 0.4,
            "understandable": 0.45,

            # ================================================================
            # Positive words - FACILITIES / ICT
            # ================================================================
            "fast": 0.4,
            "quick": 0.35,
            "high_speed": 0.55,
            "modern": 0.5,
            "well_equipped": 0.6,
            "functional": 0.4,
            "spacious": 0.45,
            "well_ventilated": 0.5,
            "well_lit": 0.4,
            "secure": 0.5,
            "safe": 0.5,
            "peaceful": 0.6,
            "quiet": 0.35,
            "serene": 0.6,

            # ================================================================
            # Negative words - GENERAL (EXPANDED)
            # ================================================================
            "terrible": -0.9,
            "horrible": -0.9,
            "frustrating": -0.8,
            "useless": -0.8,
            "poor": -0.6,
            "disappointing": -0.7,
            "annoying": -0.6,
            "bad": -0.5,
            "worst": -0.85,
            "worse": -0.7,
            "awful": -0.85,
            "dreadful": -0.8,
            "pathetic": -0.75,
            "ridiculous": -0.65,
            "absurd": -0.6,
            "unacceptable": -0.75,
            "unreasonable": -0.6,
            "unfair": -0.6,
            "unprofessional": -0.7,
            "incompetent": -0.7,
            "irresponsible": -0.7,
            "negligent": -0.65,
            "careless": -0.55,
            "lazy": -0.5,
            "slow": -0.6,
            "late": -0.4,
            "delayed": -0.45,
            "inefficient": -0.5,
            "disorganized": -0.5,
            "messy": -0.45,
            "dirty": -0.5,
            "unclean": -0.5,
            "unhygienic": -0.65,
            "unsanitary": -0.7,
            "overcrowded": -0.55,
            "crowded": -0.35,
            "noisy": -0.35,
            "disruptive": -0.5,
            "rude": -0.6,
            "arrogant": -0.6,
            "dishonest": -0.7,
            "corrupt": -0.8,
            "unhelpful": -0.55,
            "unresponsive": -0.5,
            "ignored": -0.6,
            "neglected": -0.6,
            "abandoned": -0.65,
            "abysmal": -0.85,
            "atrocious": -0.9,
            "appalling": -0.85,
            "deplorable": -0.85,
            "disgusting": -0.8,
            "revolting": -0.8,
            "repulsive": -0.8,
            "repugnant": -0.8,
            "nauseating": -0.75,
            "sickening": -0.75,
            "vile": -0.85,
            "wretched": -0.8,
            "miserable": -0.75,
            "dismal": -0.7,
            "dire": -0.7,
            "grim": -0.6,
            "bleak": -0.55,
            "gloomy": -0.5,
            "dreary": -0.5,
            "somber": -0.4,
            "melancholy": -0.5,
            "despondent": -0.6,
            "despairing": -0.65,
            "desperate": -0.6,
            "hopeless": -0.7,
            "pessimistic": -0.5,
            "cynical": -0.45,
            "skeptical": -0.3,
            "dubious": -0.4,
            "suspicious": -0.45,
            "questionable": -0.4,
            "doubtful": -0.4,
            "uncertain": -0.3,
            "unsure": -0.25,
            "hesitant": -0.25,
            "reluctant": -0.3,
            "unwilling": -0.35,
            "resistant": -0.4,
            "hostile": -0.6,
            "antagonistic": -0.6,
            "belligerent": -0.65,
            "combative": -0.6,
            "aggressive": -0.55,
            "violent": -0.7,
            "brutal": -0.75,
            "cruel": -0.75,
            "harsh": -0.55,
            "severe": -0.5,
            "stern": -0.4,
            "strict": -0.3,
            "rigid": -0.35,
            "inflexible": -0.4,
            "unyielding": -0.45,
            "stubborn": -0.45,
            "obstinate": -0.5,
            "recalcitrant": -0.55,
            "defiant": -0.5,
            "rebellious": -0.45,
            "insubordinate": -0.5,
            "disobedient": -0.45,
            "unruly": -0.45,
            "unmanageable": -0.5,
            "uncontrollable": -0.55,
            "chaotic": -0.6,
            "turbulent": -0.5,
            "tumultuous": -0.5,
            "troubled": -0.5,
            "troublesome": -0.55,
            "bothersome": -0.45,
            "irritating": -0.5,
            "irksome": -0.45,
            "vexing": -0.5,
            "galling": -0.5,
            "infuriating": -0.65,
            "maddening": -0.6,
            "exasperating": -0.6,
            "infuriating": -0.65,
            "outrageous": -0.7,
            "scandalous": -0.7,
            "shameful": -0.65,
            "disgraceful": -0.7,
            "ignominious": -0.7,
            "infamous": -0.6,
            "notorious": -0.55,
            "disreputable": -0.55,
            "discreditable": -0.6,
            "dishonorable": -0.65,
            "disgraceful": -0.7,
            "shameful": -0.65,
            "humiliating": -0.65,
            "degrading": -0.6,
            "demeaning": -0.55,
            "belittling": -0.5,
            "disparaging": -0.5,
            "derogatory": -0.55,
            "contemptuous": -0.6,
            "scornful": -0.55,
            "disdainful": -0.55,
            "contemptible": -0.6,
            "despicable": -0.7,
            "detestable": -0.7,
            "abhorrent": -0.75,
            "loathsome": -0.7,
            "odious": -0.7,
            "obnoxious": -0.6,
            "offensive": -0.55,
            "insulting": -0.55,
            "abusive": -0.65,
            "vituperative": -0.65,
            "vitriolic": -0.65,
            "venomous": -0.65,
            "malicious": -0.65,
            "malevolent": -0.7,
            "maleficent": -0.7,
            "malignant": -0.65,
            "pernicious": -0.6,
            "noxious": -0.55,
            "toxic": -0.6,
            "poisonous": -0.65,
            "venomous": -0.65,
            "corrosive": -0.55,
            "caustic": -0.5,
            "acerbic": -0.45,
            "acrimonious": -0.55,
            "acrid": -0.45,
            "bitter": -0.4,
            "acerbic": -0.45,
            "astringent": -0.4,
            "harsh": -0.5,
            "sharp": -0.35,
            "keen": -0.25,
            "cutting": -0.4,
            "incisive": -0.3,
            "trenchant": -0.35,
            "caustic": -0.5,
            "sarcastic": -0.45,
            "sardonic": -0.45,
            "ironic": -0.3,
            "cynical": -0.45,
            "mocking": -0.5,
            "sneering": -0.5,
            "scoffing": -0.5,
            "jeering": -0.5,
            "taunting": -0.5,
            "ridiculing": -0.55,
            "deriding": -0.55,
            "mocking": -0.5,

            # ================================================================
            # Negative words - ACADEMIC (EXPANDED)
            # ================================================================
            "boring": -0.4,
            "confusing": -0.45,
            "difficult": -0.35,
            "hard": -0.3,
            "stressful": -0.6,
            "overwhelming": -0.5,
            "tedious": -0.4,
            "monotonous": -0.35,
            "irrelevant": -0.45,
            "outdated": -0.5,
            "disorganized": -0.5,
            "unprepared": -0.5,
            "unqualified": -0.6,
            "biased": -0.55,
            "unfair": -0.6,
            "unclear": -0.35,
            "vague": -0.35,
            "incomplete": -0.45,
            "absent": -0.6,
            "absenteeism": -0.65,
            "truancy": -0.6,
            "negligence": -0.65,
            "dereliction": -0.7,
            "abandonment": -0.65,
            "disappearance": -0.5,
            "missing": -0.45,
            "unavailable": -0.5,
            "unreachable": -0.45,
            "inaccessible": -0.5,
            "non_responsive": -0.55,
            "unresponsive": -0.55,
            "silent": -0.35,
            "unanswered": -0.5,
            "ignored": -0.6,
            "overlooked": -0.5,
            "neglected": -0.6,
            "forgotten": -0.45,
            "disregarded": -0.55,
            "dismissed": -0.5,
            "rejected": -0.55,
            "denied": -0.5,
            "refused": -0.5,
            "declined": -0.45,
            "unfulfilled": -0.55,
            "unmet": -0.5,
            "unsatisfied": -0.55,
            "displeased": -0.6,
            "discontent": -0.55,
            "dissatisfied": -0.6,
            "disgruntled": -0.65,
            "frustrated": -0.7,
            "exasperated": -0.65,
            "infuriated": -0.75,
            "outraged": -0.8,
            "appalled": -0.75,
            "dismayed": -0.65,
            "disheartened": -0.6,
            "discouraged": -0.55,
            "disappointed": -0.7,
            "let_down": -0.6,
            "failed": -0.65,
            "failure": -0.7,
            "fiasco": -0.75,
            "debacle": -0.75,
            "disaster": -0.8,
            "catastrophe": -0.85,
            "calamity": -0.85,
            "tragedy": -0.8,
            "misfortune": -0.6,
            "mishap": -0.45,
            "setback": -0.5,
            "obstacle": -0.45,
            "hindrance": -0.45,
            "impediment": -0.45,
            "barrier": -0.4,
            "hurdle": -0.35,
            "difficulty": -0.4,
            "problem": -0.5,
            "issue": -0.4,
            "concern": -0.35,
            "worry": -0.45,
            "trouble": -0.5,
            "predicament": -0.5,
            "quandary": -0.45,
            "dilemma": -0.45,
            "plight": -0.5,
            "hardship": -0.55,
            "adversity": -0.55,
            "suffering": -0.65,
            "distress": -0.6,
            "anguish": -0.7,
            "agony": -0.75,
            "torment": -0.7,
            "torture": -0.8,
            "ordeal": -0.65,
            "nightmare": -0.75,
            "horror": -0.75,
            "terror": -0.8,
            "panic": -0.7,
            "alarm": -0.55,
            "dread": -0.6,
            "fear": -0.55,
            "fright": -0.55,
            "scare": -0.45,
            "shock": -0.5,
            "trauma": -0.7,
            "hurt": -0.55,
            "pain": -0.55,
            "ache": -0.45,
            "sore": -0.35,
            "suffering": -0.65,
            "grief": -0.7,
            "sorrow": -0.65,
            "sadness": -0.6,
            "unhappiness": -0.6,
            "misery": -0.7,
            "woe": -0.65,
            "heartache": -0.65,
            "heartbreak": -0.7,
            "despair": -0.75,
            "depression": -0.75,
            "anxiety": -0.6,
            "stress": -0.55,
            "tension": -0.45,
            "pressure": -0.4,
            "burden": -0.5,
            "load": -0.3,
            "weight": -0.35,
            "strain": -0.45,
            "drain": -0.45,
            "exhaustion": -0.55,
            "fatigue": -0.45,
            "tiredness": -0.4,
            "weariness": -0.45,
            "burnout": -0.65,

            # ================================================================
            # Negative words - FACILITIES / ICT
            # ================================================================
            "broken": -0.6,
            "damaged": -0.55,
            "faulty": -0.55,
            "defective": -0.6,
            "malfunctioning": -0.55,
            "unreliable": -0.55,
            "unstable": -0.5,
            "unavailable": -0.5,
            "offline": -0.5,
            "disconnected": -0.45,
            "lagging": -0.4,
            "buffering": -0.35,
            "congested": -0.4,
            "insufficient": -0.5,
            "inadequate": -0.55,
            "limited": -0.3,
            "restricted": -0.35,
            "shortage": -0.45,
            "lack": -0.4,
            "outage": -0.55,
            "blackout": -0.6,
            "flooded": -0.65,
            "leaking": -0.5,
            "collapsed": -0.7,
            "deteriorating": -0.55,
            "run_down": -0.5,
            "vandalized": -0.7,
            "graffiti": -0.4,

            # ================================================================
            # Catering / Food
            # ================================================================
            "delicious": 0.7,
            "tasty": 0.6,
            "nutritious": 0.5,
            "fresh": 0.5,
            "hygienic": 0.5,
            "appetizing": 0.55,
            "flavorful": 0.6,
            "bland": -0.5,
            "stale": -0.55,
            "undercooked": -0.6,
            "overcooked": -0.5,
            "overpriced": -0.5,
            "expensive": -0.35,
            "cold": -0.3,
            "spoiled": -0.7,
            "rotten": -0.8,
            "expired": -0.7,
            "contaminated": -0.75,
            "poisonous": -0.85,

            # ================================================================
            # Implicit-negative complaint phrases - GENERAL
            # These carry no single strongly negative word on their own
            # (e.g. "high cost", "still not fixed") but are consistently
            # negative framings in student feedback. Multi-word phrases
            # are matched as underscore-joined substrings against the
            # already-cleaned text (lowercased, punctuation stripped).
            # ================================================================
            "high_cost": -0.4,
            "high_fee": -0.4,
            "high_fees": -0.4,
            "high_price": -0.4,
            "high_prices": -0.4,
            "too_costly": -0.45,
            "cost_too_much": -0.45,
            "cost_a_lot": -0.4,
            "can_t_afford": -0.6,
            "cannot_afford": -0.6,
            "unaffordable": -0.55,
            "no_response": -0.4,
            "no_reply": -0.4,
            "still_not_fixed": -0.55,
            "not_yet_fixed": -0.5,
            "nothing_has_been_done": -0.55,
            "nothing_is_being_done": -0.55,
            "no_one_cares": -0.6,
            "no_one_is_listening": -0.55,
            "waste_of_money": -0.6,
            "waste_of_time": -0.55,
            "not_worth_it": -0.5,
            "not_worth_the_money": -0.55,

            # ================================================================
            # Hostel / Accommodation
            # ================================================================
            "comfortable": 0.5,
            "cosy": 0.5,
            "spacious": 0.45,
            "well_maintained": 0.55,
            "habitable": 0.35,
            "plumbing": -0.3,
            "sewage": -0.6,
            "drainage": -0.4,
            "cockroach": -0.6,
            "rodent": -0.65,
            "infestation": -0.75,
            "mold": -0.55,
            "mildew": -0.5,
            "damp": -0.4,
            "musty": -0.45,
            "ventilation": 0.3,
            "stuffy": -0.35,
            "overcrowding": -0.55,
            "congested": -0.4,
            "unsafe": -0.75,
            "hazardous": -0.7,

            # ================================================================
            # Safety / Security
            # ================================================================
            # Kidnapping / abduction
            "kidnap": -0.95,
            "kidnapped": -0.95,
            "kidnapping": -0.95,
            "kidnapper": -0.9,
            "kidnappers": -0.9,
            "abduct": -0.9,
            "abducted": -0.9,
            "abduction": -0.9,
            "hostage": -0.95,
            "hostages": -0.95,
            # Violent / weapons / shooting
            "gunshots": -0.95,
            "gunshot": -0.95,
            "shooting": -0.9,
            "shoot": -0.8,
            "shot": -0.7,
            "weapon": -0.9,
            "firearm": -0.9,
            "gun": -0.7,
            "shooter": -0.9,
            # Stabbing / assault / robbery
            "stabbing": -0.9,
            "stabbed": -0.9,
            "stab": -0.85,
            "assault": -0.9,
            "assaulted": -0.9,
            "raped": -0.95,
            "sexual_assault": -0.95,
            "robbery": -0.8,
            "robbed": -0.85,
            "mugged": -0.8,
            "mugging": -0.8,
            # Explosives
            "bomb": -0.9,
            "explosion": -0.8,
            "explode": -0.8,
            "exploded": -0.8,
            # Threats / attacks
            "threatened": -0.75,
            "threatening": -0.75,
            "attacked": -0.85,
            "ambushed": -0.85,
            "intruder": -0.75,
            "intruders": -0.75,
            # Safety concerns (not necessarily critical incidents)
            "violence": -0.85,
            "violent": -0.8,
            "threat": -0.8,
            "threatens": -0.7,
            "harassment": -0.8,
            "harassed": -0.75,
            "harassing": -0.75,
            "armed": -0.85,
            "attack": -0.85,
            "blood": -0.8,
            "injury": -0.8,
            "injured": -0.85,
            "injure": -0.85,
            "danger": -0.8,
            "dangerous": -0.75,
            "hazard": -0.7,
            "theft": -0.7,
            "stolen": -0.7,
            "stole": -0.85,
            "theft": -0.7,
            "robbery": -0.8,
            "robbed": -0.85,
            "maintenance": -0.3,
            "need_better": -0.5,
            "needs_better": -0.5,
            "need_maintenance": -0.5,
            "needs_maintenance": -0.5,
            "better_maintenance": -0.5,
            "burglary": -0.8,
            "break_in": -0.75,
            "trespassing": -0.65,
            "vandalism": -0.7,
            "stalker": -0.8,
            "lurking": -0.65,
            "suspicious": -0.5,
            "creepy": -0.55,
            "menacing": -0.7,
            "abuse": -0.7,
            "cctv": 0.35,
            "security": 0.3,
            "patrol": 0.3,
            "fire": -0.65,
            "crash": -0.6,

            # ================================================================
            # Student Wellbeing / Mental Health
            # ================================================================
            "stressed": -0.6,
            "anxious": -0.55,
            "depressed": -0.75,
            "overwhelmed": -0.5,
            "burned_out": -0.65,
            "exhausted": -0.45,
            "tired": -0.3,
            "homesick": -0.5,
            "lonely": -0.55,
            "isolated": -0.55,
            "worried": -0.45,
            "scared": -0.5,
            "afraid": -0.5,
            "nervous": -0.35,
            "pressured": -0.45,
            "struggling": -0.45,
            "suffering": -0.65,
            "counseling": 0.4,
            "support": 0.5,
            "wellness": 0.5,
            "wellbeing": 0.5,
            "mental_health": 0.3,
            "therapy": 0.4,

            # ================================================================
            # Ghanaian University Slang
            # ================================================================
            "borla": -0.7,
            "yawa": -0.6,
            "chale": 0.1,  # neutral term of address
            "paa": 0.5,    # "very" or "hard" - positive when emphasizing good
            "koraa": -0.1, # "even/at all" - neutral, context-dependent
            "wai": 0.0,  # Changed from -0.2 to avoid substring matching issues
            "nyansapo": 0.5,   # wisdom
            "sika": 0.3,   # money
            "sakawa": -0.6,    # internet fraud
            "abro": 0.4,   # foreign/fancy
            "trotro": -0.2,    # public transport (neutral/negative depending)
            "obroni": 0.1,     # foreigner (neutral)
            "odie": 0.0,       # neutral slang
            "akpeteshie": -0.3, # local alcohol
            "alata": 0.1,      # Nigerian (neutral)
            "fa": 0.0,         # take
            "bra": 0.1,        # come
            "ko": 0.0,         # go
            "hwe": 0.0,        # look
            "tee": 0.0,        # wait
            "dabi": -0.1,      # no
            "yoo": 0.2,        # okay / alright
            "eye": 0.3,        # it's good
            "won": 0.0,        # see
            "dey": 0.0,        # is/are (neutral)
            "wote": 0.0,       # shut up (negative in context)
            "kwasia": -0.5,    # fool
            "gbandi": -0.3,    # clumsy
            "ngl": 0.0,        # not gonna lie (neutral)
            "fia": 0.0,        # push/force (neutral/negative)

            # ================================================================
            # Intensifiers & Modifiers
            # ================================================================
            "very": 0.15,
            "extremely": 0.0,
            "absolutely": 0.0,
            "completely": 0.15,
            "totally": 0.15,
            "highly": 0.2,
            "seriously": 0.15,
            "barely": -0.1,
            "hardly": -0.1,
            "slightly": -0.05,

            # ================================================================
            # Negations (used for phrase-level context, flagged here)
            # ================================================================
            "not": 0.0,
            "no": 0.0,
            "never": -0.3,
            "nobody": -0.2,
            "nothing": -0.2,
            "nowhere": -0.2,
            "neither": -0.15,
            "nor": -0.1,
            "cannot": -0.1,
            "can't": -0.1,
            "don't": -0.1,
            "doesn't": -0.1,
            "didn't": -0.1,
            "won't": -0.15,
            "wouldn't": -0.1,
            "shouldn't": -0.1,
            "isn't": -0.1,
            "aren't": -0.1,
            "wasn't": -0.1,
            "weren't": -0.1,
            "hasn't": -0.1,
            "haven't": -0.1,
            "hadn't": -0.1,

            # ================================================================
            # Campus-specific locations / facilities
            # ================================================================
            "sarbah": 0.0,
            "commonwealth": 0.0,
            "republic": 0.0,
            "katanga": 0.0,
            "jubilee": 0.0,
            "ghana_house": 0.0,
            "south_block": 0.0,
            "pentagon": 0.0,
            "presidential": 0.0,
            "conti": 0.0,
            "stevenson": 0.0,
            "tetteh": 0.0,
            "q_block": 0.0,
            "business_school": 0.0,
            "engineering": 0.0,
            "science_lab": 0.0,
            "computer_lab": 0.0,
            "library": 0.2,
            "auditorium": 0.1,
            "cafeteria": -0.1,
            "dining_hall": -0.1,
            "front_desk": 0.0,
            "registry": -0.1,
            "finance_office": -0.1,
            "health_center": 0.2,
            "clinic": 0.2,
            "sports_complex": 0.3,
            "gym": 0.3,
            "basketball_court": 0.2,
            "volleyball_court": 0.2,
            "football_field": 0.2,
            "car_park": 0.0,
            "gate": 0.0,
            "security_post": 0.1,

            # ================================================================
            # Sarcasm & Irony Detection Patterns (EXPANDED)
            # These phrases indicate sarcasm - flip the sentiment
            # ================================================================
            "learned_the_true_meaning_of_patience": -0.7,
            "true_meaning_of_patience": -0.65,
            "waiting_for_a_miracle": -0.7,
            "waiting_for_miracle": -0.7,
            "forgotten_how_to_work": -0.6,
            "technically_working": -0.5,
            "would_not_exactly_call": -0.5,
            "not_exactly_reliable": -0.55,
            "not_something_i_look_forward_to": -0.6,
            "different_story": -0.5,
            "fine_until_you_need_it": -0.55,
            "fine_until": -0.45,
            "fine_with_the_portal_until": -0.55,
            "fine_but": -0.4,
            "good_but": -0.4,
            "great_but": -0.4,
            "excellent_but": -0.4,
            "okay_but": -0.4,
            "nice_but": -0.4,
            "wonderful_but": -0.4,
            "perfect_but": -0.4,
            "impressive_but": -0.4,
            "satisfied_but": -0.4,
            "pleased_but": -0.4,
            "happy_but": -0.4,
            "content_but": -0.4,
            "comfortable_but": -0.4,
            "convenient_but": -0.4,
            "affordable_but": -0.4,
            "fast_but": -0.4,
            "reliable_but": -0.4,
            "efficient_but": -0.4,
            "effective_but": -0.4,
            "professional_but": -0.4,
            "helpful_but": -0.4,
            "friendly_but": -0.4,
            "courteous_but": -0.4,
            "polite_but": -0.4,
            "respectful_but": -0.4,
            "supportive_but": -0.4,
            "responsive_but": -0.4,
            "timely_but": -0.4,
            "prompt_but": -0.4,
            "consistent_but": -0.4,
            "stable_but": -0.4,
            "clean_but": -0.4,
            "tidy_but": -0.4,
            "organized_but": -0.4,
            "spacious_but": -0.4,
            "modern_but": -0.4,
            "well_equipped_but": -0.4,
            "functional_but": -0.4,
            "well_maintained_but": -0.4,
            "habitable_but": -0.4,
            "peaceful_but": -0.4,
            "quiet_but": -0.4,
            "serene_but": -0.4,
            "safe_but": -0.4,
            "secure_but": -0.4,
            "bright_but": -0.4,
            "well_ventilated_but": -0.4,
            "well_lit_but": -0.4,
            "air_conditioned_but": -0.4,
            "heated_but": -0.4,
            "cool_but": -0.4,
            "warm_but": -0.4,
            "not_terrible_but": -0.5,
            "not_bad_but": -0.45,
            "not_good_but": -0.5,
            "i_am_happy_but": -0.5,
            "i_am_satisfied_but": -0.5,
            "i_appreciate_but": -0.5,
            "i_appreciate_although": -0.5,
            "although_half": -0.4,
            "although_some": -0.4,
            "although_many": -0.4,
            "but_the": -0.3,
            "but_some": -0.3,
            "but_many": -0.3,
            "but_half": -0.35,
            "but_most": -0.35,
            "but_few": -0.3,
            "but_little": -0.3,
            "but_less": -0.3,
            "but_more": -0.3,
            "but_worse": -0.4,
            "but_worst": -0.45,
            "but_bad": -0.4,
            "but_terrible": -0.5,
            "but_awful": -0.5,
            "but_horrible": -0.5,
            "but_poor": -0.4,
            "but_slow": -0.4,
            "but_expensive": -0.4,
            "but_dirty": -0.4,
            "but_broken": -0.4,
            "but_damaged": -0.4,
            "but_faulty": -0.4,
            "but_not_working": -0.45,
            "but_not_functional": -0.45,
            "but_unavailable": -0.4,
            "but_offline": -0.4,
            "but_disconnected": -0.4,
            "but_unstable": -0.4,
            "but_unreliable": -0.4,
            "but_inconsistent": -0.4,
            "but_insufficient": -0.4,
            "but_inadequate": -0.4,
            "but_limited": -0.35,
            "but_restricted": -0.35,
            "but_shortage": -0.4,
            "but_lack": -0.4,
            "but_outage": -0.45,
            "but_blackout": -0.5,
            "but_flooded": -0.5,
            "but_leaking": -0.45,
            "but_collapsed": -0.55,
            "but_deteriorating": -0.45,
            "but_run_down": -0.4,
            "but_vandalized": -0.5,
            "but_graffiti": -0.35,
            "comfortable_until": -0.5,
            "good_until": -0.5,
            "fine_until": -0.5,
            "okay_until": -0.5,
            "great_until": -0.5,
            "excellent_until": -0.5,
            "perfect_until": -0.5,
            "wonderful_until": -0.5,
            "amazing_until": -0.5,
            "fantastic_until": -0.5,
            "impressive_until": -0.5,
            "satisfied_until": -0.5,
            "pleased_until": -0.5,
            "happy_until": -0.5,
            "content_until": -0.5,
            "comfortable_until": -0.5,
            "convenient_until": -0.5,
            "affordable_until": -0.5,
            "fast_until": -0.5,
            "reliable_until": -0.5,
            "efficient_until": -0.5,
            "effective_until": -0.5,
            "professional_until": -0.5,
            "helpful_until": -0.5,
            "friendly_until": -0.5,
            "courteous_until": -0.5,
            "polite_until": -0.5,
            "respectful_until": -0.5,
            "supportive_until": -0.5,
            "responsive_until": -0.5,
            "timely_until": -0.5,
            "prompt_until": -0.5,
            "consistent_until": -0.5,
            "stable_until": -0.5,
            "clean_until": -0.5,
            "tidy_until": -0.5,
            "organized_until": -0.5,
            "spacious_until": -0.5,
            "modern_until": -0.5,
            "well_equipped_until": -0.5,
            "functional_until": -0.5,
            "well_maintained_until": -0.5,
            "habitable_until": -0.5,
            "peaceful_until": -0.5,
            "quiet_until": -0.5,
            "serene_until": -0.5,
            "safe_until": -0.5,
            "secure_until": -0.5,
            "bright_until": -0.5,
            "well_ventilated_until": -0.5,
            "well_lit_until": -0.5,
            "air_conditioned_until": -0.5,
            "heated_until": -0.5,
            "cool_until": -0.5,
            "warm_until": -0.5,

            # ================================================================
            # Phrase-Level Patterns (Context-Aware)
            # These multi-word phrases capture context that single words miss
            # ================================================================
            # Academic context - negative
            "lecturer_is_absent": -0.7,
            "lecturer_absent": -0.65,
            "professor_absent": -0.65,
            "teacher_absent": -0.6,
            "lecturer_not_coming": -0.65,
            "class_cancelled": -0.5,
            "class_canceled": -0.5,
            "no_lecturer": -0.6,
            "lecturer_did_not_come": -0.7,
            "lecturer_has_not_come": -0.65,
            "lecturer_is_late": -0.5,
            "lecturer_came_late": -0.55,
            "class_not_holding": -0.6,
            "course_not_taught": -0.65,
            "syllabus_not_covered": -0.6,
            "syllabus_incomplete": -0.55,
            "exam_postponed": -0.5,
            "exam_cancelled": -0.55,
            "result_not_released": -0.6,
            "grade_not_posted": -0.55,
            "assignment_not_marked": -0.6,
            "project_not_supervised": -0.65,
            "no_supervision": -0.6,
            "no_tutorial": -0.55,
            "tutorial_cancelled": -0.5,
            "lab_not_open": -0.55,
            "practical_cancelled": -0.55,
            "no_notes": -0.5,
            "no_handout": -0.45,
            "no_teaching": -0.65,
            "poor_teaching": -0.6,
            "bad_teaching": -0.65,
            "teaching_is_poor": -0.6,
            "teaching_is_bad": -0.65,
            "not_teaching_well": -0.6,
            "cannot_teach": -0.65,
            "does_not_explain": -0.6,
            "explanation_not_clear": -0.55,
            "voice_not_audible": -0.5,
            "cannot_hear": -0.45,
            "too_fast": -0.4,
            "rushing_syllabus": -0.55,
            "not_giving_assignments": -0.55,
            "not_marking_assignments": -0.6,
            "biased_grading": -0.65,
            "unfair_grading": -0.65,
            "harsh_marking": -0.6,
            "strict_marking": -0.45,
            "not_available_for_consultation": -0.6,
            "office_not_open": -0.5,
            "not_responding_to_emails": -0.6,
            "ignoring_students": -0.65,
            "not_caring": -0.6,
            "does_not_care": -0.6,

            # Catering/Food context - negative
            "food_is_cold": -0.6,
            "food_cold": -0.55,
            "meal_is_cold": -0.6,
            "rice_is_cold": -0.55,
            "food_is_bad": -0.65,
            "food_is_terrible": -0.75,
            "food_is_disgusting": -0.8,
            "food_not_good": -0.55,
            "food_not_fresh": -0.6,
            "food_is_stale": -0.65,
            "food_is_rotten": -0.8,
            "food_is_expired": -0.75,
            "food_is_contaminated": -0.8,
            "food_poisoning": -0.85,
            "food_not_hygienic": -0.7,
            "food_is_unhygienic": -0.7,
            "dirty_kitchen": -0.65,
            "dirty_canteen": -0.6,
            "dirty_dining": -0.6,
            "flies_in_food": -0.75,
            "worms_in_food": -0.8,
            "hair_in_food": -0.7,
            "stones_in_food": -0.65,
            "portion_small": -0.5,
            "portion_is_small": -0.5,
            "food_is_expensive": -0.55,
            "food_too_expensive": -0.6,
            "high_price": -0.45,
            "prices_are_high": -0.5,
            "not_affordable": -0.55,
            "cannot_afford": -0.6,
            "menu_is_limited": -0.4,
            "no_variety": -0.4,
            "same_food_everyday": -0.5,
            "no_vegetables": -0.4,
            "no_meat": -0.35,
            "water_not_clean": -0.6,
            "water_is_dirty": -0.65,

            # ICT/Technology context - negative
            "wifi_is_down": -0.7,
            "wifi_down": -0.65,
            "wifi_not_working": -0.65,
            "wifi_is_slow": -0.55,
            "internet_is_slow": -0.55,
            "internet_not_working": -0.65,
            "no_internet": -0.6,
            "no_wifi": -0.55,
            "network_is_down": -0.65,
            "network_down": -0.6,
            "portal_not_working": -0.6,
            "portal_is_down": -0.6,
            "cannot_login": -0.55,
            "cannot_access_portal": -0.55,
            "server_is_down": -0.65,
            "server_down": -0.6,
            "website_not_loading": -0.55,
            "page_not_loading": -0.5,
            "system_is_down": -0.6,
            "system_down": -0.55,
            "computer_not_working": -0.55,
            "computers_are_slow": -0.5,
            "no_computers": -0.5,
            "printer_not_working": -0.5,
            "projector_not_working": -0.5,
            "no_power": -0.6,
            "no_electricity": -0.6,
            "power_outage": -0.65,
            "light_not_working": -0.45,
            "fan_not_working": -0.45,
            "ac_not_working": -0.5,

            # Facilities context - negative
            "no_water": -0.65,
            "water_not_flowing": -0.6,
            "water_is_brown": -0.7,
            "water_is_dirty": -0.65,
            "no_toilet": -0.6,
            "toilet_is_dirty": -0.65,
            "toilet_not_flushing": -0.6,
            "bathroom_is_dirty": -0.6,
            "shower_not_working": -0.55,
            "no_hot_water": -0.5,
            "room_is_dirty": -0.55,
            "room_not_clean": -0.5,
            "hostel_is_dirty": -0.6,
            "hall_is_dirty": -0.6,
            "no_bed": -0.5,
            "bed_is_broken": -0.55,
            "mattress_is_bad": -0.55,
            "no_mattress": -0.5,
            "window_is_broken": -0.5,
            "door_is_broken": -0.5,
            "lock_is_broken": -0.5,
            "no_light": -0.45,
            "light_is_dim": -0.4,
            "fan_is_noisy": -0.4,
            "room_is_hot": -0.45,
            "room_is_cold": -0.4,
            "room_is_small": -0.35,
            "overcrowded_room": -0.55,
            "too_many_students": -0.5,
            "no_ventilation": -0.5,
            "room_is_stuffy": -0.5,
            "leaking_roof": -0.6,
            "leaking_pipe": -0.55,
            "flooding": -0.65,
            "flooded": -0.65,
            "no_drainage": -0.55,
            "sewage_problem": -0.65,
            "sewage_leak": -0.65,
            "cockroach_infestation": -0.7,
            "rodent_infestation": -0.7,
            "mosquito_infestation": -0.6,
            "no_pest_control": -0.55,

            # Safety/Security context - negative
            "no_security": -0.6,
            "security_is_poor": -0.6,
            "not_safe": -0.65,
            "feeling_unsafe": -0.65,
            "felt_unsafe": -0.65,
            "was_robbed": -0.8,
            "got_robbed": -0.8,
            "phone_stolen": -0.75,
            "laptop_stolen": -0.75,
            "bag_snatched": -0.75,
            "was_attacked": -0.85,
            "got_attacked": -0.85,
            "was_assaulted": -0.9,
            "got_assaulted": -0.9,
            "was_harassed": -0.8,
            "got_harassed": -0.8,
            "was_threatened": -0.75,
            "got_threatened": -0.75,
            "dark_corner": -0.5,
            "no_lighting": -0.55,
            "street_light_not_working": -0.5,
            "no_cctv": -0.45,
            "suspicious_person": -0.55,
            "suspicious_activity": -0.55,
            "strange_noise": -0.45,
            "break_in": -0.7,
            "attempted_break_in": -0.65,
            "theft_on_campus": -0.7,
            "stealing_on_campus": -0.65,

            # Transport context - negative
            "no_bus": -0.55,
            "bus_not_coming": -0.55,
            "bus_is_late": -0.5,
            "missed_bus": -0.45,
            "no_transport": -0.55,
            "transport_is_expensive": -0.5,
            "fare_is_high": -0.45,
            "traffic_is_bad": -0.45,
            "too_much_traffic": -0.5,
            "road_is_bad": -0.5,
            "potholes": -0.45,
            "accident": -0.65,
            "car_crash": -0.7,

            # Health context - negative
            "not_feeling_well": -0.45,
            "am_sick": -0.5,
            "fell_sick": -0.5,
            "health_center_closed": -0.6,
            "clinic_closed": -0.55,
            "no_doctor": -0.6,
            "no_nurse": -0.55,
            "no_medicine": -0.6,
            "wait_time_too_long": -0.55,
            "long_queue": -0.45,
            "no_emergency_care": -0.65,

            # Administrative context - negative
            "no_response": -0.55,
            "no_reply": -0.5,
            "still_not_fixed": -0.65,
            "not_yet_fixed": -0.55,
            "nothing_has_been_done": -0.65,
            "nothing_is_being_done": -0.65,
            "no_one_cares": -0.65,
            "no_one_is_listening": -0.6,
            "waste_of_money": -0.65,
            "waste_of_time": -0.6,
            "not_worth_it": -0.55,
            "too_much_delay": -0.55,
            "delayed_too_much": -0.6,
            "waiting_for_long": -0.5,
            "long_wait": -0.45,
            "bureaucracy": -0.5,
            "too_much_protocol": -0.45,
            "corruption": -0.8,
            "bribery": -0.8,
            "nepotism": -0.7,
            "favoritism": -0.6,

            # Positive phrases
            "lecturer_is_good": 0.6,
            "lecturer_is_great": 0.7,
            "teaching_is_excellent": 0.75,
            "teaching_is_good": 0.6,
            "food_is_good": 0.55,
            "food_is_delicious": 0.7,
            "food_is_tasty": 0.6,
            "wifi_is_fast": 0.55,
            "internet_is_fast": 0.55,
            "wifi_is_good": 0.5,
            "service_is_good": 0.55,
            "service_is_excellent": 0.7,
            "staff_are_helpful": 0.6,
            "staff_are_friendly": 0.6,
            "environment_is_clean": 0.55,
            "environment_is_nice": 0.5,
            "room_is_clean": 0.5,
            "room_is_comfortable": 0.55,
            "security_is_good": 0.5,
            "security_is_tight": 0.55,
            "i_am_happy": 0.65,
            "i_am_satisfied": 0.6,
            "i_appreciate": 0.65,
            "thank_you": 0.6,
            "well_done": 0.6,
            "good_job": 0.6,
            "keep_it_up": 0.55,
            "great_work": 0.65,
            "excellent_work": 0.75,
            "amazing_service": 0.7,
            "wonderful_experience": 0.7,
            "best_school": 0.7,
            "best_decision": 0.65,
            "proud_to_be_here": 0.6,
            "love_this_place": 0.7,
            "enjoying_my_stay": 0.6,
            "happy_to_be_here": 0.65,
            "feeling_at_home": 0.55,
            "sense_of_belonging": 0.5,
            "great_community": 0.6,
            "supportive_environment": 0.6,
            "caring_staff": 0.6,
            "helpful_lecturers": 0.6,
            "knowledgeable_lecturers": 0.65,
            "inspiring_lecturers": 0.7,
            "motivating_lecturers": 0.65,
            "accessible_lecturers": 0.55,
            "responsive_admin": 0.55,
            "efficient_service": 0.55,
            "quick_response": 0.55,
            "timely_feedback": 0.5,
            "fair_treatment": 0.5,
            "transparent_process": 0.5,
            "good_facilities": 0.5,
            "clean_environment": 0.5,
            "safe_campus": 0.55,
            "peaceful_environment": 0.55,
            "convenient_location": 0.45,
            "affordable_fees": 0.5,
            "value_for_money": 0.55,
            "quality_education": 0.6,
            "excellent_academics": 0.65,
            "good_reputation": 0.55,
            "recommended_school": 0.6,
            "would_recommend": 0.55,
            "happy_student": 0.65,
            "satisfied_student": 0.6,
            "grateful_student": 0.65,
            "blessed_to_be_here": 0.6,
            "thankful_for_this_opportunity": 0.65,

        }

    def calculate_score(self, text: str) -> float:
        """
        Calculate the average sentiment score from the custom lexicon.

        Supports:
        - Individual word matching
        - Multi-word phrase matching (e.g. "well_structured", "break_in", etc.)
        - Basic negation detection (modifies score of next sentiment word)

        A number of entries in this lexicon (kidnap, assault, weapon, etc.)
        overlap with sentiment.safety_vocabulary's authoritative safety term
        lists. That module is context-aware (e.g. "the workshop discussed
        kidnapping" is not an incident); this lexicon originally was not,
        which let it independently re-trigger negativity on the same words
        even when safety_vocabulary correctly stayed silent. When the text is
        discussing/learning about a safety topic rather than reporting one,
        those specific overlapping entries are skipped here too so the two
        modules can't drift out of sync.
        """

        if not text or not text.strip():
            return 0.0

        # Phase 2: merge active admin-managed terms from the database at runtime.
        # This keeps the admin lexicon manager useful without requiring a code edit
        # or application restart. Outside a Flask application context we simply
        # fall back to the built-in lexicon.
        runtime_lexicon = self.lexicon
        try:
            from flask import has_app_context
            if has_app_context():
                from database import CustomLexicon
                runtime_lexicon = dict(self.lexicon)
                for row in CustomLexicon.query.filter_by(is_active=True).all():
                    runtime_lexicon[row.word.strip().lower()] = float(row.sentiment_score)
        except Exception:
            runtime_lexicon = self.lexicon

        # Normalize text
        text_lower = text.lower().strip()

        skip_terms = _SAFETY_VOCAB_OVERLAP if is_discussion_context(text_lower) else frozenset()

        # First, check for multi-word phrases (underscored keys in lexicon)
        words = text_lower.split()
        matched_scores = []
        matched_phrase_words = set()  # Track words that are part of matched phrases

        # Check bigrams/trigrams from the text against underscored lexicon keys
        text_with_underscores = text_lower.replace(" ", "_")
        for phrase, score in runtime_lexicon.items():
            if "_" in phrase:
                if phrase in skip_terms:
                    continue
                if phrase in text_with_underscores:
                    matched_scores.append(score)
                    # Track which words are part of this phrase
                    for w in phrase.split("_"):
                        matched_phrase_words.add(w)

        # Now check individual words with simple negation detection
        negation_window = 3  # words after a negation to flip
        negated = False
        negation_countdown = 0

        for word in words:
            # Remove punctuation
            clean_word = word.strip(".,!?;:\"'()[]{}")

            if not clean_word:
                continue

            # Skip words that are part of matched phrases
            if clean_word in matched_phrase_words:
                continue

            # Check if this word is a negation
            if clean_word in ("no", "not", "never", "neither", "nor", "nobody", "nothing", "nowhere"):
                negated = True
                negation_countdown = negation_window
                continue

            # Check for contracted negations (don't, won't, can't, isn't, etc.)
            # Note: Only match "n't" suffix, NOT bare "nt" which catches words
            # like "excellent", "important", "brilliant", "different", etc.
            if clean_word.endswith("n't"):
                negated = True
                negation_countdown = negation_window
                continue

            # Check if word is in lexicon (skip negations themselves and phrases)
            # Use word boundary check to avoid substring matches
            if clean_word in runtime_lexicon and clean_word not in skip_terms:
                # Additional check: ensure the word is not a substring of a matched phrase
                is_substring = False
                for phrase_word in matched_phrase_words:
                    if clean_word != phrase_word and clean_word in phrase_word:
                        is_substring = True
                        break
                
                if not is_substring:
                    word_score = runtime_lexicon[clean_word]

                    # Apply negation: flip the score if within negation window
                    if negated and negation_countdown > 0 and abs(word_score) >= 0.3:
                        word_score = -word_score

                    matched_scores.append(word_score)

            # Decrement negation counter
            if negation_countdown > 0:
                negation_countdown -= 1
                if negation_countdown == 0:
                    negated = False

        # If no custom word was found
        if not matched_scores:
            return 0.0

        # Calculate the average score
        # Weight negative scores more heavily (complaints are more important)
        negative_scores = [s for s in matched_scores if s < 0]
        positive_scores = [s for s in matched_scores if s > 0]
        
        if negative_scores and positive_scores:
            # Mixed sentiment - use the most extreme score (complaints prioritized)
            most_negative = min(negative_scores)
            most_positive = max(positive_scores)
            # If there's a strong negative phrase, trust it
            if abs(most_negative) >= 0.5:
                final_score = most_negative
            else:
                # Weight negative more heavily
                final_score = (sum(negative_scores) * 2 + sum(positive_scores)) / (len(negative_scores) * 2 + len(positive_scores))
        elif negative_scores:
            final_score = sum(negative_scores) / len(negative_scores)
        elif positive_scores:
            final_score = sum(positive_scores) / len(positive_scores)
        else:
            # All scores are 0
            final_score = 0.0

        return round(final_score, 4)

    def learn_from_correction(self, text, corrected_sentiment, admin_name=None, max_terms=4):
        """Active-learning: seed the database-backed lexicon with words that the
        generic engines did not recognise, scored by the human reviewer's label.

        This closes the loop between the AI Review Queue and the custom lexicon:
        every reviewed feedback item whose sentiment the admin confirms/corrects
        to a clear polarity (Positive/Negative) teaches the system the domain
        vocabulary it was missing, so future feedback on those words is scored
        correctly. Neutral labels carry no polarity, so they are skipped.

        Words already present in the built-in or database lexicon are ignored
        (so we never overwrite a tuned term), and only alphabetic, lowercase,
        multi-character tokens are considered to avoid injecting hall names or
        leetspeak artifacts. Returns the number of terms seeded.

        Note: rows are added to the current DB session but not committed here --
        the caller is expected to commit (kept in the same transaction).
        """
        if not text or corrected_sentiment not in ("positive", "negative"):
            return 0

        # Lexicon writes require a Flask app context (database session). Called
        # outside one (e.g. tests), bail out safely and do nothing.
        try:
            from flask import has_app_context
            if not has_app_context():
                return 0
        except Exception:
            return 0

        target = 0.5 if corrected_sentiment == "positive" else -0.5

        try:
            from sentiment.unknown_detector import UnknownWordDetector
            detector = UnknownWordDetector()
            unknown = detector.detect(text)
        except Exception:
            return 0

        if not unknown:
            return 0

        # Prioritise the most frequently used unknown words in this feedback.
        freq = {}
        for w in re.findall(r"\b[a-zA-Z]+\b", str(text).lower()):
            if w in unknown:
                freq[w] = freq.get(w, 0) + 1
        candidates = sorted(unknown, key=lambda w: freq.get(w, 0), reverse=True)

        try:
            from database import db, CustomLexicon
        except Exception:
            return 0

        added = 0
        for word in candidates:
            if added >= max_terms:
                break
            if len(word) < 4 or not word.isalpha() or word[0].isupper():
                continue
            if CustomLexicon.query.filter_by(word=word).first():
                continue
            try:
                db.session.add(CustomLexicon(
                    word=word,
                    sentiment_score=target,
                    category="auto",
                    is_active=True,
                    added_by=(admin_name or "auto-learn"),
                ))
                added += 1
            except Exception:
                continue

        return added