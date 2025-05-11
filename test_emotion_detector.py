import time
from emotion_detector import EmotionDetector
import numpy as np

def test_emotion_detector():
    detector = EmotionDetector(smoothing_window=5, chunk_size=0.25, confidence_threshold=0.5)
    print("Emotion detector initialized. Starting test...")
    
    # Simulate audio input (silence)
    dummy_audio = np.zeros(int(0.25 * 16000), dtype=np.float32)
    
    # Process the dummy audio
    timestamp, emotion, confidence = detector.process_audio(dummy_audio)
    print(f"Test result: Emotion={emotion}, Confidence={confidence:.2f}")
    
    # Clean up
    detector.stop_processing()

if __name__ == "__main__":
    test_emotion_detector() 