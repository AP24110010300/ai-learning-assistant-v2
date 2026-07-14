"""
Lightweight Emotion Detection — No TensorFlow or PyTorch required.

Two classifiers:
1. NaiveBayesEmotionClassifier: scikit-learn TF-IDF + MultinomialNB trained on built-in data
2. KeywordEmotionAnalyzer: Rule-based keyword matching with weighted scoring

Both produce 5 emotion classes: Bored, Confident, Confused, Curious, Frustrated
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from preprocessing import clean_text, get_keyword_scores

EMOTION_CLASSES = ["Bored", "Confident", "Confused", "Curious", "Frustrated"]

# ─── Built-in Training Dataset ───────────────────────────────────────────────
# ~200 labeled examples covering realistic student learning scenarios.
# This trains in < 0.5 seconds and requires zero model files on disk.

TRAINING_DATA = [
    # ── Bored ──
    ("this lecture is so boring I can barely stay awake", "Bored"),
    ("i'm so tired of doing the same exercises over and over", "Bored"),
    ("this topic is not interesting at all", "Bored"),
    ("i feel sleepy every time i open this textbook", "Bored"),
    ("why do we even need to learn this it seems pointless", "Bored"),
    ("the material is so dull i can't focus", "Bored"),
    ("i keep yawning during this class", "Bored"),
    ("this is tedious and repetitive", "Bored"),
    ("i don't care about this subject anymore", "Bored"),
    ("the homework is monotonous and uninteresting", "Bored"),
    ("i have zero motivation to study this", "Bored"),
    ("every chapter feels the same nothing new", "Bored"),
    ("i just want this class to end already", "Bored"),
    ("this is a waste of my time honestly", "Bored"),
    ("i am completely uninterested in this topic", "Bored"),
    ("another boring assignment great", "Bored"),
    ("i feel exhausted just thinking about studying this", "Bored"),
    ("the professor's explanations put me to sleep", "Bored"),
    ("none of this is engaging or stimulating", "Bored"),
    ("i would rather do anything else than study this", "Bored"),
    ("this class drags on forever", "Bored"),
    ("i can't bring myself to care about these problems", "Bored"),
    ("reading this material is like watching paint dry", "Bored"),
    ("i zone out every time i try to study", "Bored"),
    ("the assignments are so repetitive and dull", "Bored"),
    ("nothing about this course excites me", "Bored"),
    ("i feel no energy to work on this project", "Bored"),
    ("it feels like we covered this already multiple times", "Bored"),
    ("my mind wanders whenever i try to focus on this", "Bored"),
    ("i just go through the motions without really learning", "Bored"),

    # ── Confident ──
    ("i totally understand this concept it makes perfect sense", "Confident"),
    ("this is easy i got this no problem", "Confident"),
    ("i feel prepared for the exam", "Confident"),
    ("i nailed the practice problems", "Confident"),
    ("the solution is clear and straightforward", "Confident"),
    ("i mastered this topic already", "Confident"),
    ("i can explain this to others now", "Confident"),
    ("this concept clicked right away", "Confident"),
    ("i feel ready for the test", "Confident"),
    ("i understand the underlying principles well", "Confident"),
    ("piece of cake this is simple", "Confident"),
    ("i solved all the exercises without any help", "Confident"),
    ("my understanding is solid on this topic", "Confident"),
    ("i can see how everything connects now", "Confident"),
    ("i feel great about my progress", "Confident"),
    ("the material is clear and well organized", "Confident"),
    ("i know exactly how to approach this problem", "Confident"),
    ("i aced the last quiz on this", "Confident"),
    ("this makes complete sense to me now", "Confident"),
    ("i have a strong grasp of the fundamentals", "Confident"),
    ("i can do this with my eyes closed", "Confident"),
    ("everything is clicking into place", "Confident"),
    ("i practiced enough and feel very prepared", "Confident"),
    ("this topic is my strongest area", "Confident"),
    ("i feel comfortable with all the key concepts", "Confident"),
    ("i can apply this knowledge to real problems", "Confident"),
    ("the homework was straightforward and easy", "Confident"),
    ("i understand both the theory and practice", "Confident"),
    ("i feel like an expert on this subject now", "Confident"),
    ("no doubts at all i understand everything", "Confident"),

    # ── Confused ──
    ("i don't understand this at all", "Confused"),
    ("this concept makes no sense to me", "Confused"),
    ("i'm completely lost in this chapter", "Confused"),
    ("can someone explain how this works", "Confused"),
    ("i'm stuck on this problem and have no idea what to do", "Confused"),
    ("the explanation is unclear and confusing", "Confused"),
    ("i keep reading but nothing clicks", "Confused"),
    ("why does this formula work i don't get it", "Confused"),
    ("this is way too complicated for me", "Confused"),
    ("i can't figure out what went wrong in my solution", "Confused"),
    ("i need help understanding the basics", "Confused"),
    ("the more i study the more confused i get", "Confused"),
    ("what does this term even mean", "Confused"),
    ("i'm struggling to follow the logic here", "Confused"),
    ("the textbook explanation is hard to follow", "Confused"),
    ("i have so many questions about this topic", "Confused"),
    ("how is this related to what we learned before", "Confused"),
    ("i don't see the connection between these concepts", "Confused"),
    ("this problem is puzzling me", "Confused"),
    ("i tried multiple approaches but nothing works", "Confused"),
    ("the instructions are not clear at all", "Confused"),
    ("i understand the words but not the meaning", "Confused"),
    ("can you break this down into simpler steps", "Confused"),
    ("i feel completely overwhelmed by the material", "Confused"),
    ("every time i think i get it something else confuses me", "Confused"),
    ("i can't wrap my head around this concept", "Confused"),
    ("the notation is really confusing me", "Confused"),
    ("i don't know where to even start", "Confused"),
    ("this feels impossibly complex", "Confused"),
    ("i read the chapter three times and still don't understand", "Confused"),

    # ── Curious ──
    ("this is really interesting i want to learn more", "Curious"),
    ("i wonder how this applies in the real world", "Curious"),
    ("this is fascinating can you tell me more", "Curious"),
    ("what if we approach this differently", "Curious"),
    ("i'm eager to explore this topic further", "Curious"),
    ("how does this connect to other fields", "Curious"),
    ("this is so cool i want to dig deeper", "Curious"),
    ("i'm intrigued by the possibilities here", "Curious"),
    ("what are the advanced applications of this", "Curious"),
    ("i want to understand the theory behind this", "Curious"),
    ("this opens up so many interesting questions", "Curious"),
    ("i'm excited to explore more examples", "Curious"),
    ("why does this phenomenon happen", "Curious"),
    ("how did scientists discover this", "Curious"),
    ("i want to try implementing this myself", "Curious"),
    ("are there any cutting edge developments in this area", "Curious"),
    ("this sparks so many ideas in my mind", "Curious"),
    ("i find this topic amazingly interesting", "Curious"),
    ("what would happen if we changed this variable", "Curious"),
    ("i love learning about this kind of stuff", "Curious"),
    ("can we go beyond what the textbook covers", "Curious"),
    ("i want to do extra research on this", "Curious"),
    ("this is the most exciting topic in the course", "Curious"),
    ("i wonder what else i can build with this knowledge", "Curious"),
    ("the possibilities are endless this is awesome", "Curious"),
    ("how does this work under the hood", "Curious"),
    ("i stayed up late reading about this because it's so interesting", "Curious"),
    ("what are some unsolved problems in this area", "Curious"),
    ("i want to write a paper about this topic", "Curious"),
    ("this reminds me of something cool i read about", "Curious"),

    # ── Frustrated ──
    ("i keep getting errors and nothing works", "Frustrated"),
    ("this is so frustrating i want to give up", "Frustrated"),
    ("i've been stuck on this for hours and made no progress", "Frustrated"),
    ("why is this so hard it should be simple", "Frustrated"),
    ("i hate this subject it makes me angry", "Frustrated"),
    ("my code keeps breaking and i don't know why", "Frustrated"),
    ("i'm so annoyed that i can't solve this", "Frustrated"),
    ("this is impossible nobody can understand this", "Frustrated"),
    ("i'm about to give up on this assignment", "Frustrated"),
    ("debugging this is a nightmare", "Frustrated"),
    ("i feel stupid because i can't figure this out", "Frustrated"),
    ("every time i fix one thing something else breaks", "Frustrated"),
    ("this problem is driving me crazy", "Frustrated"),
    ("i've tried everything and nothing works", "Frustrated"),
    ("i'm so irritated with this homework", "Frustrated"),
    ("the deadline is tomorrow and i'm nowhere close to done", "Frustrated"),
    ("i can't believe how difficult this is", "Frustrated"),
    ("i want to throw my laptop out the window", "Frustrated"),
    ("ugh this keeps failing over and over", "Frustrated"),
    ("i'm so angry at myself for not getting this", "Frustrated"),
    ("this assignment is terrible and unfair", "Frustrated"),
    ("the error messages make no sense to me", "Frustrated"),
    ("i feel like giving up on this entire course", "Frustrated"),
    ("nothing i do seems to work correctly", "Frustrated"),
    ("this is the worst topic ever", "Frustrated"),
    ("i spent the whole weekend and still can't solve it", "Frustrated"),
    ("my solution was wrong again for the third time", "Frustrated"),
    ("i'm fed up with this class", "Frustrated"),
    ("the professor didn't explain this well and now i'm stuck", "Frustrated"),
    ("i keep making the same mistakes over and over", "Frustrated"),
]


class NaiveBayesEmotionClassifier:
    """Lightweight ML classifier using TF-IDF + Multinomial Naive Bayes.
    Trains from built-in data in ~0.3 seconds. No model files needed."""

    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=3000,
                ngram_range=(1, 2),
                stop_words='english'
            )),
            ('clf', MultinomialNB(alpha=0.1))
        ])
        self.is_trained = False
        self._train()

    def _train(self):
        """Train the classifier on the built-in dataset."""
        texts = [clean_text(t) for t, _ in TRAINING_DATA]
        labels = [l for _, l in TRAINING_DATA]

        self.pipeline.fit(texts, labels)
        self.is_trained = True

    def predict(self, text):
        """Predict emotion with confidence scores for all classes."""
        if not self.is_trained:
            return None

        cleaned = clean_text(text)
        if not cleaned:
            cleaned = "none"

        # Get probability distribution across all classes
        proba = self.pipeline.predict_proba([cleaned])[0]
        class_names = self.pipeline.classes_

        # Build base scores from ML model
        base_scores = {name: prob for name, prob in zip(class_names, proba)}

        # Ensure all emotion classes are represented
        for ec in EMOTION_CLASSES:
            if ec not in base_scores:
                base_scores[ec] = 0.0

        # Add keyword enhancement
        keyword_scores = get_keyword_scores(cleaned)
        final_scores = {}
        for emotion in EMOTION_CLASSES:
            score = (base_scores[emotion] * 100) + keyword_scores[emotion]
            final_scores[emotion] = score

        # Normalize to sum to 1.0
        total = sum(final_scores.values())
        if total > 0:
            for k in final_scores:
                final_scores[k] /= total
        else:
            for k in final_scores:
                final_scores[k] = 1.0 / len(EMOTION_CLASSES)

        top_emotion = max(final_scores, key=final_scores.get)
        confidence = final_scores[top_emotion]

        return {
            'emotion': top_emotion,
            'confidence': confidence,
            'scores': final_scores,
            'cleaned_text': cleaned
        }


class KeywordEmotionAnalyzer:
    """Rule-based emotion analyzer using weighted keyword matching.
    No training needed — works instantly."""

    def __init__(self):
        self.model = True  # Flag for compatibility with app.py model checks

    def predict(self, text):
        """Predict emotion based on keyword presence and density."""
        cleaned = clean_text(text)
        if not cleaned:
            cleaned = "none"

        keyword_scores = get_keyword_scores(cleaned)

        # Add sentiment heuristics
        exclamation_count = cleaned.count('!')
        question_count = cleaned.count('?')

        # Questions often indicate confusion or curiosity
        if question_count >= 2:
            keyword_scores["Confused"] += 5
            keyword_scores["Curious"] += 3

        # Exclamations often indicate frustration or excitement
        if exclamation_count >= 2:
            keyword_scores["Frustrated"] += 3
            keyword_scores["Curious"] += 2

        # Negative sentiment words
        negative_words = ["not", "don't", "can't", "won't", "no", "never", "nothing"]
        neg_count = sum(1 for w in negative_words if w in cleaned.split())
        if neg_count >= 2:
            keyword_scores["Frustrated"] += 5
            keyword_scores["Confused"] += 3

        # Positive sentiment words
        positive_words = ["love", "great", "good", "excellent", "amazing", "wonderful"]
        pos_count = sum(1 for w in positive_words if w in cleaned.split())
        if pos_count >= 1:
            keyword_scores["Confident"] += 5
            keyword_scores["Curious"] += 3

        # If no keywords matched at all, default to Confused (most common student state)
        total_raw = sum(keyword_scores.values())
        if total_raw == 0:
            keyword_scores["Confused"] = 30
            keyword_scores["Curious"] = 25
            keyword_scores["Frustrated"] = 20
            keyword_scores["Bored"] = 15
            keyword_scores["Confident"] = 10

        # Normalize
        total = sum(keyword_scores.values())
        final_scores = {}
        for emotion in EMOTION_CLASSES:
            final_scores[emotion] = keyword_scores[emotion] / total if total > 0 else 0.2

        top_emotion = max(final_scores, key=final_scores.get)
        confidence = final_scores[top_emotion]

        return {
            'emotion': top_emotion,
            'confidence': confidence,
            'scores': final_scores,
            'cleaned_text': cleaned
        }


def get_mixed_emotions(scores, threshold=0.15):
    """Detects multiple emotions above a certain confidence threshold."""
    mixed = []
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Always include top emotion
    mixed.append(sorted_scores[0])

    # Include additional emotions above threshold
    for i in range(1, len(sorted_scores)):
        if sorted_scores[i][1] >= threshold:
            mixed.append(sorted_scores[i])

    return mixed
