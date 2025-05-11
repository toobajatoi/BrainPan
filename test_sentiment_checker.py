from sentiment_checker import SentimentChecker

checker = SentimentChecker()

if getattr(checker, 'english_only', False):
    print("WARNING: Only English sentiment analysis is available. Other languages will be treated as English or may return neutral.")

examples = [
    "I am very happy today!",      # English
    "Je suis très triste.",        # French
    "میں بہت خوش ہوں۔",              # Urdu
    "kia kar rehay ho",            # Roman Urdu
    "Estoy enojado contigo.",      # Spanish
]

for text in examples:
    sentiment, confidence, emotion, lang, warning = checker.analyze_sentiment(text)
    print(f"Text: {text}")
    print(f"Sentiment: {sentiment:.2f}, Confidence: {confidence:.2f}, Emotion: {emotion}, Language: {lang}, Warning: {warning}")
    print("-" * 40) 