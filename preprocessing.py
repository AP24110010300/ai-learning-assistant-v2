import re

# Emotion Keywords for enhancement
EMOTION_KEYWORDS = {
    "Bored": [
        "bored", "tired", "sleepy", "uninterested", "dull", "tedious",
        "monotonous", "yawn", "exhausted", "boring", "repetitive",
        "pointless", "waste of time", "don't care", "not interested"
    ],
    "Confident": [
        "confident", "easy", "got this", "simple", "understand", "clear",
        "ready", "prepared", "know", "mastered", "ace", "nailed",
        "no problem", "makes sense", "piece of cake", "straightforward"
    ],
    "Confused": [
        "confused", "lost", "stuck", "what", "how", "why",
        "don't understand", "unclear", "puzzled", "makes no sense",
        "no idea", "help", "struggling", "can't figure", "complicated",
        "complex", "hard to follow", "don't get it"
    ],
    "Curious": [
        "curious", "interesting", "fascinating", "wonder", "want to know",
        "intrigued", "eager", "explore", "learn more", "excited",
        "amazed", "cool", "awesome", "wow", "tell me more",
        "how does", "what if", "why does"
    ],
    "Frustrated": [
        "frustrated", "angry", "annoyed", "stupid", "hate", "impossible",
        "giving up", "irritated", "fail", "terrible", "awful", "ugh",
        "broken", "doesn't work", "keep getting errors", "wrong",
        "can't believe", "so hard", "nightmare"
    ]
}


def clean_text(text):
    """Cleans input text by removing special characters but preserving emotion-carrying punctuation."""
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    # Remove special characters except common punctuation that carries emotion (!, ?, .)
    text = re.sub(r'[^a-zA-Z0-9\s!?.,\']', '', text)

    return text.strip()


def get_keyword_scores(text):
    """Calculates keyword-based emotion scores with weighted matching."""
    scores = {emotion: 0.0 for emotion in EMOTION_KEYWORDS}
    text_lower = text.lower()

    for emotion, keywords in EMOTION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                scores[emotion] += 10.0  # Strong weight for explicit keywords

    return scores
