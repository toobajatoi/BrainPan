import time
import numpy as np
from fast_emotion_detector import FastEmotionDetector
from sentiment_checker import SentimentChecker
from tone_switcher import ToneSwitcher
from datetime import datetime
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from tabulate import tabulate

load_dotenv("config.env")

# Define test cases with expected emotions and sentiments
TEST_CASES = [
    {"text": "I am very happy today!", "expected_emotion": "hap", "expected_sentiment": "positive"},
    {"text": "This is terrible news.", "expected_emotion": "sad", "expected_sentiment": "negative"},
    {"text": "I'm feeling okay, just normal.", "expected_emotion": "neu", "expected_sentiment": "neutral"},
    {"text": "Wow, that's amazing!", "expected_emotion": "hap", "expected_sentiment": "positive"},
    {"text": "I'm a bit sad about the results.", "expected_emotion": "sad", "expected_sentiment": "negative"},
    {"text": "I'm so excited about this!", "expected_emotion": "hap", "expected_sentiment": "positive"},
    {"text": "This is frustrating and annoying.", "expected_emotion": "ang", "expected_sentiment": "negative"},
    {"text": "Everything is fine, no worries.", "expected_emotion": "neu", "expected_sentiment": "neutral"}
]

def plot_results(results):
    """Plot the test results"""
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot 1: Emotion Distribution
    emotions = [r['emotion'] for r in results]
    emotion_counts = {e: emotions.count(e) for e in set(emotions)}
    ax1.bar(emotion_counts.keys(), emotion_counts.values())
    ax1.set_title('Emotion Distribution')
    ax1.set_ylabel('Count')
    
    # Plot 2: Latency Trends
    test_numbers = range(1, len(results) + 1)
    ax2.plot(test_numbers, [r['emotion_latency'] for r in results], 'b-', label='Emotion')
    ax2.plot(test_numbers, [r['sentiment_latency'] for r in results], 'r-', label='Sentiment')
    ax2.plot(test_numbers, [r['tone_latency'] for r in results], 'g-', label='Tone')
    ax2.set_title('Latency Trends')
    ax2.set_xlabel('Test Number')
    ax2.set_ylabel('Latency (ms)')
    ax2.legend()
    
    # Plot 3: Confidence Scores
    ax3.plot(test_numbers, [r['emotion_conf'] for r in results], 'b-', label='Emotion')
    ax3.plot(test_numbers, [r['sentiment_conf'] for r in results], 'r-', label='Sentiment')
    ax3.set_title('Confidence Scores')
    ax3.set_xlabel('Test Number')
    ax3.set_ylabel('Confidence')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('test_results.png')
    plt.close()

def main():
    detector = FastEmotionDetector()
    checker = SentimentChecker()
    switcher = ToneSwitcher()

    results = []
    print("\n--- Automated Test Suite (FastEmotionDetector) ---\n")
    
    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"\nTest Case {i}:")
        print(f"Text: '{test_case['text']}'")
        print(f"Expected: Emotion={test_case['expected_emotion']}, Sentiment={test_case['expected_sentiment']}")
        
        # Record audio
        audio = np.random.randn(int(detector.chunk_size * detector.sample_rate)).astype(np.float32)  # Simulate audio
        t0 = time.time()
        timestamp, emotion, emo_conf = detector.process_audio(audio)
        t1 = time.time()
        emotion_latency = (t1 - t0) * 1000
        print(f"  Emotion: {emotion} (conf: {emo_conf:.2f}) | Latency: {emotion_latency:.2f} ms | Timestamp: {timestamp:.2f}")

        # Sentiment
        t2 = time.time()
        sentiment_result, sent_conf, _, _, _ = checker.analyze_sentiment(test_case['text'])
        t3 = time.time()
        sentiment_latency = (t3 - t2) * 1000
        print(f"  Sentiment: {sentiment_result['sentiment']} (score: {sentiment_result['score']:.2f}) (conf: {sent_conf:.2f}) | Latency: {sentiment_latency:.2f} ms")

        # Tone switch
        t4 = time.time()
        tone = switcher.determine_tone(emotion, emo_conf, sentiment_result['score'], sent_conf)
        t5 = time.time()
        tone_latency = (t5 - t4) * 1000
        print(f"  Tone: {tone} | Latency: {tone_latency:.2f} ms")

        # Accuracy check
        emotion_correct = emotion == test_case['expected_emotion']
        sentiment_correct = sentiment_result['sentiment'] == test_case['expected_sentiment']
        print(f"  Emotion correct? {'YES' if emotion_correct else 'NO'}")
        print(f"  Sentiment correct? {'YES' if sentiment_correct else 'NO'}")

        results.append({
            "emotion": emotion,
            "emotion_conf": emo_conf,
            "emotion_latency": emotion_latency,
            "sentiment": sentiment_result['sentiment'],
            "sentiment_score": sentiment_result['score'],
            "sentiment_conf": sent_conf,
            "sentiment_latency": sentiment_latency,
            "tone": tone,
            "tone_latency": tone_latency,
            "emotion_correct": emotion_correct,
            "sentiment_correct": sentiment_correct
        })
        
        time.sleep(1)  # Small pause between tests

    # Generate plots
    plot_results(results)

    # Summarize
    print("\n--- Summary ---\n")
    avg_emo_lat = np.mean([r["emotion_latency"] for r in results])
    avg_sent_lat = np.mean([r["sentiment_latency"] for r in results])
    avg_tone_lat = np.mean([r["tone_latency"] for r in results])
    
    emotion_accuracy = sum(1 for r in results if r["emotion_correct"]) / len(results) * 100
    sentiment_accuracy = sum(1 for r in results if r["sentiment_correct"]) / len(results) * 100
    
    print(f"Average Emotion Tag Delay: {avg_emo_lat:.2f} ms")
    print(f"Average Sentiment Delay: {avg_sent_lat:.2f} ms")
    print(f"Average Tone Switch Reaction: {avg_tone_lat:.2f} ms")
    print(f"Emotion Detection Accuracy: {emotion_accuracy:.1f}%")
    print(f"Sentiment Analysis Accuracy: {sentiment_accuracy:.1f}%")
    
    print("\nDetailed Results:")
    headers = ["Test", "Emotion", "Emo Conf", "Emo Lat", "Sentiment", "Sent Conf", "Sent Lat", "Tone", "Tone Lat"]
    table_data = []
    for i, r in enumerate(results, 1):
        table_data.append([
            i,
            f"{r['emotion']} ({'✓' if r['emotion_correct'] else '✗'})",
            f"{r['emotion_conf']:.2f}",
            f"{r['emotion_latency']:.1f}ms",
            f"{r['sentiment']} ({'✓' if r['sentiment_correct'] else '✗'})",
            f"{r['sentiment_conf']:.2f}",
            f"{r['sentiment_latency']:.1f}ms",
            r['tone'],
            f"{r['tone_latency']:.1f}ms"
        ])
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    print("\nResults visualization saved to 'test_results.png'")

if __name__ == "__main__":
    main() 