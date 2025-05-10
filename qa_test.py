import os
import time
from emotion_detector import EmotionDetector
from sentiment_checker import SentimentChecker
from tone_switcher import ToneSwitcher
from typing import List, Tuple, Dict
import numpy as np
from tabulate import tabulate

def process_audio_file(file_path: str, emotion_detector: EmotionDetector, 
                      sentiment_checker: SentimentChecker, tone_switcher: ToneSwitcher) -> Dict:
    """Process a single audio file and measure latencies"""
    try:
        # Process audio file
        start_time = time.time()
        emotion_results = emotion_detector.process_wav_file(file_path)
        emotion_latency = time.time() - start_time
        
        # Get emotion with highest confidence
        if emotion_results:
            # emotion_results: List[(i, emotion, conf)]
            best = max(emotion_results, key=lambda x: x[2])
            emotion, confidence = best[1], best[2]
        else:
            emotion, confidence = "neutral", 0.0
            
        # Process sentiment
        start_time = time.time()
        sentiment, sentiment_conf = sentiment_checker.analyze_sentiment("Test transcript")
        sentiment_latency = time.time() - start_time
        
        # Determine tone
        start_time = time.time()
        tone = tone_switcher.determine_tone(emotion, confidence, sentiment, sentiment_conf)
        tone_latency = time.time() - start_time
        
        return {
            'file': os.path.basename(file_path),
            'emotion': emotion,
            'emotion_conf': confidence,
            'emotion_latency': emotion_latency * 1000,  # Convert to ms
            'sentiment': sentiment,
            'sentiment_conf': sentiment_conf,
            'sentiment_latency': sentiment_latency * 1000,  # Convert to ms
            'tone': tone,
            'tone_latency': tone_latency * 1000  # Convert to ms
        }
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def main():
    # Initialize components
    emotion_detector = EmotionDetector()
    sentiment_checker = SentimentChecker()
    tone_switcher = ToneSwitcher()
    
    # Test files
    test_files = [
        'test_angry.wav',
        'test_happy.wav',
        'test_sad.wav',
        'test_neutral.wav',
        'test_mixed.wav'
    ]
    
    # Process each file
    results = []
    for file in test_files:
        if os.path.exists(file):
            result = process_audio_file(file, emotion_detector, sentiment_checker, tone_switcher)
            if result:
                results.append(result)
    
    # Print results
    if results:
        # Calculate averages
        avg_emotion_latency = np.mean([r['emotion_latency'] for r in results])
        avg_sentiment_latency = np.mean([r['sentiment_latency'] for r in results])
        avg_tone_latency = np.mean([r['tone_latency'] for r in results])
        
        # Print detailed results
        print("\nDetailed Results:")
        headers = ['File', 'Emotion', 'Conf', 'Latency(ms)', 'Sentiment', 'Conf', 'Latency(ms)', 'Tone', 'Latency(ms)']
        table_data = [[
            r['file'],
            r['emotion'],
            f"{r['emotion_conf']:.2f}",
            f"{r['emotion_latency']:.1f}",
            f"{r['sentiment']:.2f}",
            f"{r['sentiment_conf']:.2f}",
            f"{r['sentiment_latency']:.1f}",
            r['tone'],
            f"{r['tone_latency']:.1f}"
        ] for r in results]
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
        
        # Print summary
        print("\nLatency Summary:")
        print(f"Average Emotion Detection Latency: {avg_emotion_latency:.1f}ms")
        print(f"Average Sentiment Analysis Latency: {avg_sentiment_latency:.1f}ms")
        print(f"Average Tone Switching Latency: {avg_tone_latency:.1f}ms")
        
        # Check if latencies meet requirements
        print("\nLatency Requirements:")
        print(f"Emotion Detection < 50ms: {'✓' if avg_emotion_latency < 50 else '✗'}")
        print(f"Sentiment Analysis < 100ms: {'✓' if avg_sentiment_latency < 100 else '✗'}")
        print(f"Tone Switching < 100ms: {'✓' if avg_tone_latency < 100 else '✗'}")
    else:
        print("No results to display")

if __name__ == "__main__":
    main() 