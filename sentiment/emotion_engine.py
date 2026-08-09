"""
emotion_engine.py

Detects emotions in student feedback using an expanded
emotion keyword dictionary with intensity weighting.

Emotions detected:
- anger, fear, sadness, joy, trust, disgust, surprise
- anticipation, optimism, love, gratitude, confusion
"""


class NRCEmotionAnalyzer:

    def __init__(self):

        # Emotion words with intensity weights
        # Weight: 1.0 (mild) to 3.0 (intense)
        self.emotion_words = {

            "anger": {
                "annoyed": 1.0,
                "irritated": 1.5,
                "aggravated": 1.5,
                "frustrated": 2.0,
                "frustrating": 2.0,
                "angry": 2.0,
                "livid": 2.5,
                "enraged": 3.0,
                "furious": 3.0,
                "outraged": 3.0,
                "infuriated": 3.0,
                "fuming": 2.5,
                "bitter": 2.0,
                "resentful": 2.0,
                "hostile": 2.5,
                "rage": 3.0,
                "wrath": 3.0,
                "mad": 1.5,
                "upset": 1.5,
                "pissed": 2.0,
            },

            "fear": {
                "worried": 1.5,
                "anxious": 1.5,
                "nervous": 1.0,
                "concerned": 1.0,
                "uneasy": 1.5,
                "afraid": 2.0,
                "scared": 2.0,
                "terrified": 3.0,
                "frightened": 2.5,
                "panicked": 2.5,
                "panic": 2.5,
                "terror": 3.0,
                "horrified": 3.0,
                "dread": 2.5,
                "fearful": 2.0,
                "unsafe": 2.5,
                "dangerous": 2.0,
                "danger": 2.5,
                "threat": 2.5,
                "threatened": 2.5,
                "intimidated": 2.0,
                "helpless": 2.0,
                "vulnerable": 1.5,
                "gunshots": 3.0,
                "gunshot": 3.0,
                "shooting": 3.0,
                "shoot": 3.0,
                "shot": 2.5,
                "shooter": 3.0,
                "weapon": 3.0,
                "firearm": 3.0,
                "gun": 2.5,
                "violence": 3.0,
                "violent": 3.0,
                "attack": 3.0,
                "attacked": 3.0,
                "armed": 3.0,
                "hostage": 3.0,
                "hostages": 3.0,
                "explosion": 3.0,
                "explode": 3.0,
                "exploded": 3.0,
                "bomb": 3.0,
                "crash": 2.5,
                "suspicious": 1.5,
                "creepy": 1.5,
                # Kidnapping / abduction
                "kidnap": 3.0,
                "kidnapped": 3.0,
                "kidnapping": 3.0,
                "kidnapper": 3.0,
                "kidnappers": 3.0,
                "abduct": 3.0,
                "abducted": 3.0,
                "abduction": 3.0,
                # Stabbing / assault / robbery
                "stabbing": 3.0,
                "stabbed": 3.0,
                "stab": 3.0,
                "assault": 3.0,
                "assaulted": 3.0,
                "raped": 3.0,
                "robbery": 3.0,
                "robbed": 3.0,
                "mugged": 3.0,
                "mugging": 3.0,
                # Threatening / intruders
                "threatened": 2.5,
                "ambushed": 3.0,
                "intruder": 2.5,
                "intruders": 2.5,
            },

            "sadness": {
                "sad": 1.5,
                "unhappy": 1.5,
                "disappointed": 2.0,
                "disappointing": 2.0,
                "depressed": 3.0,
                "depressing": 2.5,
                "miserable": 2.5,
                "heartbroken": 3.0,
                "gloomy": 2.0,
                "sorrowful": 2.5,
                "grief": 3.0,
                "mournful": 2.5,
                "hopeless": 2.5,
                "despair": 3.0,
                "devastated": 3.0,
                "crushed": 2.5,
                "lonely": 2.0,
                "isolated": 2.0,
                "homesick": 1.5,
                "melancholy": 2.0,
                "tearful": 2.0,
                "crying": 2.0,
                "hurt": 2.0,
                "painful": 2.0,
                "suffering": 2.5,
                "tired": 1.0,
                "exhausted": 1.5,
                "burned_out": 2.0,
                "overwhelmed": 2.0,
            },

            "joy": {
                "happy": 2.0,
                "glad": 1.5,
                "delighted": 2.5,
                "joyful": 2.5,
                "cheerful": 2.0,
                "ecstatic": 3.0,
                "elated": 2.5,
                "thrilled": 2.5,
                "excited": 2.0,
                "excellent": 2.0,
                "amazing": 2.5,
                "wonderful": 2.5,
                "great": 1.5,
                "fantastic": 2.5,
                "brilliant": 2.0,
                "outstanding": 2.5,
                "superb": 2.5,
                "awesome": 2.0,
                "perfect": 2.0,
                "lovely": 1.5,
                "pleased": 1.5,
                "satisfied": 1.5,
                "blessed": 2.0,
                "grateful": 2.0,
                "thankful": 2.0,
                "proud": 2.0,
                "celebrating": 2.0,
                "euphoric": 3.0,
                "blissful": 2.5,
                "radiant": 2.0,
                "overjoyed": 3.0,
            },

            "trust": {
                "reliable": 2.0,
                "helpful": 1.5,
                "honest": 2.5,
                "trusted": 2.5,
                "trustworthy": 2.5,
                "efficient": 1.5,
                "dependable": 2.0,
                "consistent": 1.5,
                "transparent": 2.0,
                "accountable": 2.0,
                "responsible": 1.5,
                "professional": 1.5,
                "supportive": 2.0,
                "responsive": 1.5,
                "fair": 1.5,
                "just": 2.0,
                "integrity": 2.5,
                "loyal": 2.0,
                "committed": 1.5,
                "dedicated": 1.5,
                "assuring": 1.5,
                "confident": 1.5,
                "hopeful": 1.5,
                "optimistic": 1.5,
            },

            "disgust": {
                "disgusting": 3.0,
                "disgusted": 3.0,
                "repulsive": 2.5,
                "revolting": 2.5,
                "nauseating": 2.5,
                "sickening": 2.5,
                "terrible": 2.5,
                "horrible": 2.5,
                "awful": 2.0,
                "dreadful": 2.5,
                "useless": 2.0,
                "worthless": 2.5,
                "pathetic": 2.0,
                "gross": 2.0,
                "unhygienic": 2.0,
                "unsanitary": 2.0,
                "filthy": 2.5,
                "nasty": 2.0,
                "rotten": 2.5,
                "spoiled": 2.0,
                "contaminated": 2.5,
                "stale": 1.5,
                "bland": 1.0,
                "messy": 1.0,
                "dirty": 1.5,
                "unclean": 1.5,
            },

            "surprise": {
                "surprised": 2.0,
                "unexpected": 2.0,
                "shocking": 2.5,
                "shocked": 2.5,
                "astonishing": 2.5,
                "astonished": 2.5,
                "stunned": 2.0,
                "amazed": 2.0,
                "astounding": 2.5,
                "startling": 2.0,
                "bewildered": 1.5,
                "dumbfounded": 2.0,
                "speechless": 1.5,
                "incredible": 2.0,
                "unbelievable": 2.0,
                "remarkable": 1.5,
                "extraordinary": 2.0,
                "impressive": 1.5,
            },

            "anticipation": {
                "expecting": 1.0,
                "hope": 1.5,
                "hoping": 1.5,
                "looking_forward": 2.0,
                "awaiting": 1.0,
                "eager": 1.5,
                "anticipating": 1.5,
                "waiting": 0.5,
                "expect": 1.0,
                "expectation": 1.0,
                "prospective": 1.0,
                "soon": 0.5,
                "pending": 0.5,
                "upcoming": 0.5,
                "foresee": 1.0,
                "predict": 1.0,
            },

            "optimism": {
                "optimistic": 2.0,
                "positive": 1.5,
                "hopeful": 1.5,
                "encouraged": 1.5,
                "encouraging": 1.5,
                "promising": 2.0,
                "bright": 1.5,
                "improving": 1.5,
                "better": 1.0,
                "progress": 1.5,
                "confident": 1.5,
                "assured": 1.0,
                "solutions": 1.5,
                "resolve": 1.5,
                "fixable": 1.0,
                "resolvable": 1.0,
            },

            "love": {
                "love": 2.5,
                "loved": 2.5,
                "adore": 2.5,
                "cherish": 2.5,
                "appreciate": 2.0,
                "appreciation": 2.0,
                "grateful": 2.0,
                "thankful": 2.0,
                "affection": 2.0,
                "care": 1.5,
                "caring": 1.5,
                "fond": 1.5,
                "dear": 1.5,
                "precious": 2.0,
                "beloved": 2.5,
                "warm": 1.0,
                "kindness": 1.5,
                "compassion": 2.0,
            },

            "gratitude": {
                "thank": 2.0,
                "thanks": 2.0,
                "thankful": 2.0,
                "grateful": 2.5,
                "appreciate": 2.0,
                "appreciation": 2.0,
                "indebted": 2.0,
                "blessed": 2.0,
                "gratitude": 2.5,
                "acknowledge": 1.0,
                "recognize": 1.0,
                "commend": 1.5,
                "praise": 1.5,
            },

            "confusion": {
                "confused": 1.5,
                "confusing": 1.5,
                "unclear": 1.5,
                "vague": 1.0,
                "ambiguous": 1.5,
                "uncertain": 1.0,
                "unclear": 1.5,
                "puzzled": 1.5,
                "perplexed": 1.5,
                "baffled": 2.0,
                "bewildered": 1.5,
                "lost": 1.5,
                "don't_understand": 2.0,
                "no_idea": 1.5,
                "mystified": 1.5,
                "confounding": 1.5,
                "contradictory": 1.5,
            },
        }

    def analyze(self, text: str) -> dict:
        """
        Analyze text and return emotion scores with intensity weighting.

        Returns:
        - dominant_emotion: The strongest emotion detected
        - emotion_scores: Raw scores per emotion (sum of word intensities)
        - emotion_intensities: Normalized (0-1) intensity for each emotion
        - secondary_emotions: Other emotions with notable scores
        """

        if not text or not text.strip():
            return {
                "dominant_emotion": "neutral",
                "emotion_scores": {e: 0 for e in self.emotion_words},
                "emotion_intensities": {e: 0.0 for e in self.emotion_words},
                "secondary_emotions": [],
                "compound_mood": "neutral"
            }

        # Convert the feedback to lowercase words
        words = text.lower().split()

        # Start every emotion at zero
        emotion_scores = {
            emotion: 0
            for emotion in self.emotion_words
        }

        # Intensity-weighted scoring
        for word in words:
            # Remove punctuation
            clean_word = word.strip(".,!?;:'\"()[]{}")

            for emotion, keywords in self.emotion_words.items():
                if clean_word in keywords:
                    # Add the intensity weight (not just +1)
                    emotion_scores[emotion] += keywords[clean_word]

        # Find the emotion with the highest score
        if max(emotion_scores.values()) == 0:
            dominant_emotion = "neutral"
        else:
            dominant_emotion = max(
                emotion_scores,
                key=emotion_scores.get
            )

        # Calculate normalized intensities (0 to 1)
        max_score = max(emotion_scores.values()) if emotion_scores else 0
        emotion_intensities = {}
        if max_score > 0:
            for emotion, score in emotion_scores.items():
                emotion_intensities[emotion] = round(score / max_score, 3)
        else:
            emotion_intensities = {e: 0.0 for e in emotion_scores}

        # Find secondary emotions (score > 0 and not dominant)
        secondary = []
        if dominant_emotion != "neutral":
            sorted_emotions = sorted(
                emotion_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            for emotion, score in sorted_emotions:
                if emotion != dominant_emotion and score > 0:
                    secondary.append(emotion)
                    if len(secondary) >= 2:
                        break

        # Determine compound mood based on emotion blend
        compound_mood = self._determine_compound_mood(emotion_scores, dominant_emotion)

        return {
            "dominant_emotion": dominant_emotion,
            "emotion_scores": emotion_scores,
            "emotion_intensities": emotion_intensities,
            "secondary_emotions": secondary,
            "compound_mood": compound_mood
        }

    def _determine_compound_mood(
        self,
        emotion_scores: dict,
        dominant: str
    ) -> str:
        """
        Determine a compound mood label from blended emotions.

        Examples:
        - anger + sadness = "frustrated"
        - fear + sadness = "distressed"
        - joy + trust = "admiring"
        - anger + fear = "threatened"
        """
        anger = emotion_scores.get("anger", 0)
        fear = emotion_scores.get("fear", 0)
        sadness = emotion_scores.get("sadness", 0)
        joy = emotion_scores.get("joy", 0)
        trust = emotion_scores.get("trust", 0)
        disgust = emotion_scores.get("disgust", 0)
        surprise = emotion_scores.get("surprise", 0)

        # Check for mixed emotions
        if anger > 0 and sadness > 0 and max(anger, sadness) > 2:
            return "frustrated_resignation"
        if anger > 0 and fear > 0 and max(anger, fear) > 2:
            return "threatened_defensive"
        if fear > 0 and sadness > 0 and max(fear, sadness) > 2:
            return "distressed_anxious"
        if joy > 0 and trust > 0 and max(joy, trust) > 2:
            return "admiring_appreciative"
        if joy > 0 and surprise > 0 and max(joy, surprise) > 2:
            return "delighted_surprised"
        if disgust > 0 and anger > 0 and max(disgust, anger) > 2:
            return "revolted_indignant"
        if sadness > 0 and disgust > 0 and max(sadness, disgust) > 2:
            return "disheartened"
        if fear > 0 and surprise > 0 and max(fear, surprise) > 2:
            return "alarmed"

        # Pure dominant emotion
        dominant_moods = {
            "anger": "agitated",
            "fear": "anxious",
            "sadness": "somber",
            "joy": "cheerful",
            "trust": "confident",
            "disgust": "repulsed",
            "surprise": "astonished",
            "anticipation": "expectant",
            "optimism": "hopeful",
            "love": "affectionate",
            "gratitude": "thankful",
            "confusion": "perplexed",
        }

        if emotion_scores.get(dominant, 0) > 0:
            return dominant_moods.get(dominant, "neutral")

        return "neutral"
