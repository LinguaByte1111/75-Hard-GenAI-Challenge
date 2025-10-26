import re

# --- Define word lists and sarcasm cues ---
positive_words = {"great", "love", "awesome", "amazing", "fantastic", "wonderful", "best", "cool", "nice"}
negative_words = {"hate", "awful", "terrible", "worst", "bad", "boring", "stupid", "disaster", "annoying"}
sarcasm_indicators = [
    "yeah right", "sure", "as if", "totally", "of course", "wow just wow", "nice job", "great job", "brilliant", "not!"
]

def get_sentiment_score(text):
    """Return a rough sentiment score (+ve, -ve)."""
    words = re.findall(r"\b\w+\b", text.lower())
    pos = len([w for w in words if w in positive_words])
    neg = len([w for w in words if w in negative_words])
    return pos - neg  # >0 => positive, <0 => negative

def rule_based_sarcasm_detector(text):
    text_lower = text.lower()

    # Check for explicit sarcasm markers
    for phrase in sarcasm_indicators:
        if phrase in text_lower:
            return True, f"Sarcasm marker phrase found: '{phrase}'"

    # Check contradictory sentiment pattern
    sentiment_score = get_sentiment_score(text)
    if sentiment_score > 0 and any(word in text_lower for word in negative_words):
        return True, "Positive tone mixed with negative word(s)"
    if sentiment_score < 0 and any(word in text_lower for word in positive_words):
        return True, "Negative tone mixed with positive word(s)"

    # Check for sarcastic punctuation
    if "..." in text or ("!" in text and sentiment_score <= 0):
        return True, "Suspicious punctuation pattern (ellipses/exclamation)"

    return False, "No sarcasm detected"

# --- Take input from user ---
user_input = input("Enter a sentence: ")

# --- Analyze ---
is_sarcastic, reason = rule_based_sarcasm_detector(user_input)

# --- Output result ---
print("\nResult:")
print(f"Sarcastic: {is_sarcastic}")
print(f"Reason: {reason}")
