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


class CustomLexiconManager:

    def __init__(self):

        # Custom sentiment dictionary
        # Scores range from -1.0 (most negative) to +1.0 (most positive)
        self.lexicon = {

            # ================================================================
            # Positive words - GENERAL
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
            # Negative words - GENERAL
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
            "slow": -0.35,
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

            # ================================================================
            # Negative words - ACADEMIC
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
            "gunshots": -0.95,
            "gunshot": -0.95,
            "shooting": -0.9,
            "weapon": -0.9,
            "violence": -0.85,
            "assault": -0.9,
            "harassment": -0.8,
            "emergency": -0.8,
            "hostage": -0.95,
            "threat": -0.8,
            "threatened": -0.75,
            "armed": -0.85,
            "attack": -0.85,
            "blood": -0.8,
            "injury": -0.8,
            "injured": -0.85,
            "danger": -0.8,
            "hazard": -0.7,
            "theft": -0.7,
            "stolen": -0.7,
            "robbery": -0.8,
            "burglary": -0.8,
            "break_in": -0.75,
            "intruder": -0.75,
            "trespassing": -0.65,
            "vandalism": -0.7,
            "stalker": -0.8,
            "lurking": -0.65,
            "suspicious": -0.5,
            "creepy": -0.55,
            "cctv": 0.35,
            "security": 0.3,
            "patrol": 0.3,
            "fire": -0.65,
            "explosion": -0.8,
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
            "wai": -0.2,   # exclamation, usually frustration
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
            "extremely": 0.25,
            "absolutely": 0.2,
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

        }

    def calculate_score(self, text: str) -> float:
        """
        Calculate the average sentiment score from the custom lexicon.

        Supports:
        - Individual word matching
        - Multi-word phrase matching (e.g. "well_structured", "break_in", etc.)
        - Basic negation detection (modifies score of next sentiment word)
        """

        if not text or not text.strip():
            return 0.0

        # Normalize text
        text_lower = text.lower().strip()

        # First, check for multi-word phrases (underscored keys in lexicon)
        words = text_lower.split()
        matched_scores = []

        # Check bigrams/trigrams from the text against underscored lexicon keys
        text_with_underscores = text_lower.replace(" ", "_")
        for phrase, score in self.lexicon.items():
            if "_" in phrase:
                if phrase in text_with_underscores:
                    matched_scores.append(score)

        # Now check individual words with simple negation detection
        negation_window = 3  # words after a negation to flip
        negated = False
        negation_countdown = 0

        for word in words:
            # Remove punctuation
            clean_word = word.strip(".,!?;:\"'()[]{}")

            if not clean_word:
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
            if clean_word in self.lexicon:
                word_score = self.lexicon[clean_word]

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
        final_score = (
            sum(matched_scores)
            / len(matched_scores)
        )

        return round(final_score, 4)
