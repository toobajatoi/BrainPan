from dotenv import load_dotenv
load_dotenv("config.env")
import time
from emotion_detector import EmotionDetector
from sentiment_checker import SentimentChecker
from tone_switcher import ToneSwitcher
import threading
import queue
import sounddevice as sd
import numpy as np
from datetime import datetime

class FeelAwareSystem:
    def __init__(self):
        self.emotion_detector = EmotionDetector()
        self.sentiment_checker = SentimentChecker()
        self.tone_switcher = ToneSwitcher()
        
        self.audio_queue = queue.Queue()
        self.text_queue = queue.Queue()
        self.is_running = False
        
        # Performance metrics
        self.metrics = {
            "emotion_latency": [],
            "sentiment_latency": [],
            "tone_switch_latency": []
        }
    
    def start(self):
        """Start the feel-aware system"""
        self.is_running = True
        
        # Start audio recording thread
        self.audio_thread = threading.Thread(target=self._audio_recording_loop)
        self.audio_thread.start()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.start()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.start()
    
    def stop(self):
        """Stop the feel-aware system"""
        self.is_running = False
        self.audio_thread.join()
        self.processing_thread.join()
        self.monitoring_thread.join()
    
    def _audio_recording_loop(self):
        """Continuously record audio in 1-second chunks"""
        while self.is_running:
            audio_data = self.emotion_detector.record_audio(duration=1.0)
            self.audio_queue.put(audio_data)
            time.sleep(0.1)  # Small delay to prevent CPU overload
    
    def _processing_loop(self):
        """Process audio and text data"""
        while self.is_running:
            try:
                # Process audio for emotion
                if not self.audio_queue.empty():
                    audio_data = self.audio_queue.get()
                    emotion, emotion_conf = self.emotion_detector.process_audio(audio_data)
                    self.metrics["emotion_latency"].append(
                        self.emotion_detector.get_average_latency()
                    )
                
                # Process text for sentiment (simulated for demo)
                # In a real system, this would come from a speech-to-text service
                if not self.text_queue.empty():
                    text = self.text_queue.get()
                    sentiment, sentiment_conf = self.sentiment_checker.analyze_sentiment(text)
                    self.metrics["sentiment_latency"].append(
                        self.sentiment_checker.get_average_latency()
                    )
                    
                    # Determine tone
                    tone = self.tone_switcher.determine_tone(
                        emotion, emotion_conf,
                        sentiment, sentiment_conf
                    )
                    self.metrics["tone_switch_latency"].append(
                        self.tone_switcher.get_average_latency()
                    )
                    
                    # Print current state
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}]")
                    print(f"Emotion: {emotion} (conf: {emotion_conf:.2f})")
                    print(f"Sentiment: {sentiment:.2f} (conf: {sentiment_conf:.2f})")
                    print(f"Current Tone: {tone}")
                    
            except Exception as e:
                print(f"Error in processing loop: {e}")
            
            time.sleep(0.1)
    
    def _monitoring_loop(self):
        """Monitor system performance"""
        while self.is_running:
            try:
                # Calculate average latencies
                avg_emotion_latency = np.mean(self.metrics["emotion_latency"][-100:]) if self.metrics["emotion_latency"] else 0
                avg_sentiment_latency = np.mean(self.metrics["sentiment_latency"][-100:]) if self.metrics["sentiment_latency"] else 0
                avg_tone_latency = np.mean(self.metrics["tone_switch_latency"][-100:]) if self.metrics["tone_switch_latency"] else 0
                
                # Print performance metrics
                print("\nPerformance Metrics:")
                print(f"Emotion Detection Latency: {avg_emotion_latency:.2f}ms")
                print(f"Sentiment Analysis Latency: {avg_sentiment_latency:.2f}ms")
                print(f"Tone Switching Latency: {avg_tone_latency:.2f}ms")
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
            
            time.sleep(5)  # Update metrics every 5 seconds

def main():
    print("Starting Feel-Aware Tone Switcher System...")
    detector = EmotionDetector()
    checker = SentimentChecker()
    switcher = ToneSwitcher()
    print("Components initialized successfully.")

    # Example: Record audio and detect emotion
    print("Recording audio for emotion detection...")
    audio = detector.record_audio(duration=1.0)
    emotion, confidence, lang = detector.process_audio(audio)
    print(f"Detected emotion: {emotion} (confidence: {confidence:.2f}, lang: {lang})")

    # Example: Analyze sentiment
    text = "I am very happy today!"
    print(f"Analyzing sentiment for text: '{text}'")
    sentiment = checker.analyze_sentiment(text)
    print(f"Sentiment analysis result: {sentiment}")

    # Example: Adjust tone and generate speech
    print("Adjusting tone and generating speech...")
    audio = switcher.adjust_tone(text, emotion, sentiment)
    if audio:
        print("Speech generated successfully.")
        # Optionally save the audio to a file
        if hasattr(audio, '__iter__') and not isinstance(audio, (bytes, bytearray)):
            audio_bytes = b''.join(audio)
        else:
            audio_bytes = audio
        with open("output.wav", "wb") as f:
            f.write(audio_bytes)
        print("Audio saved to 'output.wav'.")
    else:
        print("Failed to generate speech.")

if __name__ == "__main__":
    main() 