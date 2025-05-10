import time
import numpy as np
from emotion_detector import EmotionDetector
from sentiment_checker import SentimentChecker
from tone_switcher import ToneSwitcher
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv("config.env")

# Optionally, define expected tones for accuracy calculation
# Example: ["happy", "neutral", ...]
EXPECTED_TONES = [None, None, None, None, None]  # Set to None if you don't want to check

# Optionally, define test texts for sentiment
TEST_TEXTS = [
    "I am very happy today!",
    "This is terrible news.",
    "I'm feeling okay, just normal.",
    "Wow, that's amazing!",
    "I'm a bit sad about the results."
]

def main():
    detector = EmotionDetector()
    checker = SentimentChecker()
    switcher = ToneSwitcher()

    results = []
    print("\n--- Automated 5 Test Calls ---\n")
    for i in range(5):
        print(f"Test Call {i+1}:")
        # Record audio (or use a sample if you want)
        audio = detector.record_audio(duration=1.0)
        t0 = time.time()
        emotion, emo_conf, lang = detector.process_audio(audio)
        t1 = time.time()
        emotion_latency = (t1 - t0) * 1000
        print(f"  Emotion: {emotion} (conf: {emo_conf:.2f}, lang: {lang}) | Latency: {emotion_latency:.2f} ms")

        # Sentiment
        text = TEST_TEXTS[i % len(TEST_TEXTS)]
        t2 = time.time()
        sentiment, sent_conf = checker.analyze_sentiment(text)
        t3 = time.time()
        sentiment_latency = (t3 - t2) * 1000
        print(f"  Sentiment: {sentiment:.2f} (conf: {sent_conf:.2f}) | Latency: {sentiment_latency:.2f} ms")

        # Tone switch
        t4 = time.time()
        tone = switcher.determine_tone(emotion, emo_conf, sentiment, sent_conf)
        t5 = time.time()
        tone_latency = (t5 - t4) * 1000
        print(f"  Tone: {tone} | Latency: {tone_latency:.2f} ms")

        # Optionally, generate speech (comment out if not needed)
        # audio_out = switcher.adjust_tone(text, emotion, sentiment)
        # if audio_out:
        #     with open(f"output_{i+1}.wav", "wb") as f:
        #         f.write(audio_out if isinstance(audio_out, bytes) else b''.join(audio_out))

        # Accuracy check
        correct = None
        if EXPECTED_TONES[i] is not None:
            correct = (tone == EXPECTED_TONES[i])
            print(f"  Tone correct? {'YES' if correct else 'NO'} (expected: {EXPECTED_TONES[i]})")

        results.append({
            "emotion": emotion,
            "emotion_conf": emo_conf,
            "emotion_latency": emotion_latency,
            "sentiment": sentiment,
            "sentiment_conf": sent_conf,
            "sentiment_latency": sentiment_latency,
            "tone": tone,
            "tone_latency": tone_latency,
            "correct": correct
        })
        print()
        time.sleep(1)  # Small pause between tests

    # Summarize
    print("\n--- Summary ---\n")
    avg_emo_lat = np.mean([r["emotion_latency"] for r in results])
    avg_sent_lat = np.mean([r["sentiment_latency"] for r in results])
    avg_tone_lat = np.mean([r["tone_latency"] for r in results])
    print(f"Average Emotion Tag Delay: {avg_emo_lat:.2f} ms")
    print(f"Average Sentiment Delay: {avg_sent_lat:.2f} ms")
    print(f"Average Tone Switch Reaction: {avg_tone_lat:.2f} ms")
    if any(r["correct"] is not None for r in results):
        correct_count = sum(1 for r in results if r["correct"])
        print(f"Tone Accuracy: {correct_count}/5 correct choices")
    print("\nDetails:")
    for i, r in enumerate(results):
        print(f"Test {i+1}: Emotion={r['emotion']} ({r['emotion_conf']:.2f}), Sentiment={r['sentiment']:.2f} ({r['sentiment_conf']:.2f}), Tone={r['tone']}, Latencies: E={r['emotion_latency']:.2f}ms, S={r['sentiment_latency']:.2f}ms, T={r['tone_latency']:.2f}ms")

if __name__ == "__main__":
    main() 