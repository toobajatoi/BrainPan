import time
import numpy as np
import librosa
from emotion_detector import EmotionDetector
from sentiment_checker import SentimentChecker
from tone_switcher import ToneSwitcher
import os
from tabulate import tabulate
import matplotlib.pyplot as plt

def test_emotion_detection(detector, audio_path):
    """Test emotion detection performance"""
    print(f"\nTesting emotion detection on {audio_path}...")
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # Process in 1-second chunks
    chunk_size = int(sr)
    chunks = [audio[i:i+chunk_size] for i in range(0, len(audio), chunk_size)]
    
    results = []
    latencies = []
    
    for i, chunk in enumerate(chunks):
        start_time = time.time()
        timestamp, emotion, confidence = detector.process_audio(chunk)
        latency = (time.time() - start_time) * 1000
        latencies.append(latency)
        results.append((i, emotion, confidence, latency))
        print(f"Chunk {i}: {emotion} (conf: {confidence:.2f}) | Latency: {latency:.2f} ms")
    
    avg_latency = sum(latencies) / len(latencies)
    print(f"\nEmotion Detection Average Latency: {avg_latency:.2f} ms")
    
    return results, avg_latency

def test_sentiment_analysis(checker, text):
    """Test sentiment analysis performance"""
    print(f"\nTesting sentiment analysis on text: '{text}'...")
    
    start_time = time.time()
    sentiment, score, confidence = checker.analyze_sentiment(text)
    latency = (time.time() - start_time) * 1000
    
    print(f"Sentiment: {sentiment} (score: {score:.2f}, conf: {confidence:.2f})")
    print(f"Latency: {latency:.2f} ms")
    
    return sentiment, score, confidence, latency

def test_tone_switching(switcher, emotion, emotion_conf, sentiment, sentiment_conf):
    """Test tone switching performance"""
    print(f"\nTesting tone switching...")
    print(f"Input: Emotion={emotion} (conf={emotion_conf:.2f}), Sentiment={sentiment:.2f} (conf={sentiment_conf:.2f})")
    
    start_time = time.time()
    tone = switcher.determine_tone(emotion, emotion_conf, sentiment, sentiment_conf)
    latency = (time.time() - start_time) * 1000
    
    print(f"Output Tone: {tone}")
    print(f"Latency: {latency:.2f} ms")
    
    return tone, latency

def test_end_to_end(detector, checker, switcher, audio_path, text):
    """Test end-to-end performance"""
    print(f"\nTesting end-to-end pipeline...")
    
    # 1. Emotion Detection
    start_time = time.time()
    audio, sr = librosa.load(audio_path, sr=16000)
    timestamp, emotion, emotion_conf = detector.process_audio(audio)
    emotion_latency = (time.time() - start_time) * 1000
    
    # 2. Sentiment Analysis
    start_time = time.time()
    sentiment, sentiment_score, sentiment_conf = checker.analyze_sentiment(text)
    sentiment_latency = (time.time() - start_time) * 1000
    
    # 3. Tone Switching
    start_time = time.time()
    tone = switcher.determine_tone(emotion, emotion_conf, sentiment_score, sentiment_conf)
    tone_latency = (time.time() - start_time) * 1000
    
    # 4. Total latency
    total_latency = emotion_latency + sentiment_latency + tone_latency
    
    print("\nResults:")
    print(f"Emotion: {emotion} (conf: {emotion_conf:.2f}) | Latency: {emotion_latency:.2f} ms")
    print(f"Sentiment: {sentiment} (score: {sentiment_score:.2f}, conf: {sentiment_conf:.2f}) | Latency: {sentiment_latency:.2f} ms")
    print(f"Tone: {tone} | Latency: {tone_latency:.2f} ms")
    print(f"Total Latency: {total_latency:.2f} ms")
    
    return {
        'emotion': emotion,
        'emotion_conf': emotion_conf,
        'emotion_latency': emotion_latency,
        'sentiment': sentiment,
        'sentiment_score': sentiment_score,
        'sentiment_conf': sentiment_conf,
        'sentiment_latency': sentiment_latency,
        'tone': tone,
        'tone_latency': tone_latency,
        'total_latency': total_latency
    }

def plot_latencies(results):
    """Plot latency results"""
    components = ['Emotion Detection', 'Sentiment Analysis', 'Tone Switching']
    latencies = [
        results['emotion_latency'],
        results['sentiment_latency'],
        results['tone_latency']
    ]
    
    plt.figure(figsize=(10, 6))
    plt.bar(components, latencies)
    plt.title('Component Latencies')
    plt.ylabel('Latency (ms)')
    plt.axhline(y=50, color='r', linestyle='--', label='Target (50ms)')
    plt.legend()
    plt.savefig('latency_results.png')
    plt.close()

def main():
    # Initialize components
    detector = EmotionDetector()
    checker = SentimentChecker()
    switcher = ToneSwitcher()
    
    # Test files
    test_audio = "test_audio/angry.wav"  # Replace with your test audio
    test_text = "I am very angry about this situation!"
    
    # Run tests
    print("Starting performance tests...")
    
    # 1. Emotion Detection Test
    emotion_results, emotion_avg_latency = test_emotion_detection(detector, test_audio)
    
    # 2. Sentiment Analysis Test
    sentiment_results = test_sentiment_analysis(checker, test_text)
    
    # 3. Tone Switching Test
    tone_results = test_tone_switching(
        switcher,
        emotion_results[0][1],  # emotion
        emotion_results[0][2],  # emotion_conf
        sentiment_results[1],   # sentiment_score
        sentiment_results[2]    # sentiment_conf
    )
    
    # 4. End-to-End Test
    e2e_results = test_end_to_end(detector, checker, switcher, test_audio, test_text)
    
    # Plot results
    plot_latencies(e2e_results)
    
    # Print summary
    print("\nPerformance Summary:")
    summary = [
        ["Emotion Detection", f"{e2e_results['emotion_latency']:.2f} ms", "Target: < 50 ms"],
        ["Sentiment Analysis", f"{e2e_results['sentiment_latency']:.2f} ms", "Target: < 100 ms"],
        ["Tone Switching", f"{e2e_results['tone_latency']:.2f} ms", "Target: < 100 ms"],
        ["Total Pipeline", f"{e2e_results['total_latency']:.2f} ms", "Target: < 200 ms"]
    ]
    print(tabulate(summary, headers=["Component", "Latency", "Target"], tablefmt="grid"))

if __name__ == "__main__":
    main() 