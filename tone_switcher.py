from elevenlabs import ElevenLabs, Voice, VoiceSettings
import os
from dotenv import load_dotenv
import time
from typing import Dict, Optional, Tuple, List
from collections import deque
import threading
import queue

class ToneSwitcher:
    def __init__(self, cache_size=100, smoothing_window=3):
        load_dotenv()
        # self.api_key = os.getenv('ELEVEN_LABS_API_KEY')  # Only needed for speech
        self.latency_history = deque(maxlen=100)
        self.smoothing_window = smoothing_window
        self.recent_predictions = deque(maxlen=smoothing_window)
        self.confidence_threshold = 0.6
        self.current_tone = 'neutral'
        
        # Streaming processing setup
        self.input_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        self.is_processing = False
        self.processing_thread = None
        
        # Start processing thread
        self.start_processing()
        
        # Define tone mappings
        self.tone_mappings = {
            'neutral': {'stability': 0.7, 'similarity_boost': 0.7},
            'happy': {'stability': 0.3, 'similarity_boost': 0.8},
            'gentle': {'stability': 0.8, 'similarity_boost': 0.5},
            'calm': {'stability': 0.9, 'similarity_boost': 0.4},
            'angry': {'stability': 0.5, 'similarity_boost': 0.5}
        }
    
    def start_processing(self):
        """Start the background processing thread"""
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._process_stream)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def stop_processing(self):
        """Stop the background processing thread"""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join()

    def _process_stream(self):
        """Background thread for processing tone decisions"""
        while self.is_processing:
            try:
                if not self.input_queue.empty():
                    emotion, emotion_conf, sentiment, sentiment_conf = self.input_queue.get()
                    result = self._determine_tone_internal(emotion, emotion_conf, sentiment, sentiment_conf)
                    self.result_queue.put(result)
            except Exception as e:
                print(f"Error in processing thread: {str(e)}")
            time.sleep(0.001)  # Small sleep to prevent CPU overload

    def _determine_tone_internal(self, emotion: str, emotion_conf: float, sentiment: float, sentiment_conf: float) -> str:
        """Internal method to determine tone with minimal latency"""
        try:
            start_time = time.time()
            
            # Fast decision making
            if emotion_conf < self.confidence_threshold and sentiment_conf < self.confidence_threshold:
                tone = 'neutral'
            else:
                # Weighted decision based on confidence
                if emotion_conf > sentiment_conf:
                    # Emotion-based decision
                    if emotion == 'angry':
                        tone = 'calm'  # Counteract anger
                    elif emotion == 'happy':
                        tone = 'happy'
                    elif emotion == 'sad':
                        tone = 'gentle'
                    else:
                        tone = 'neutral'
                else:
                    # Sentiment-based decision
                    if sentiment > 0.3:
                        tone = 'happy'
                    elif sentiment < -0.3:
                        tone = 'gentle'
                    else:
                        tone = 'neutral'
            
            # Calculate latency
            latency = (time.time() - start_time) * 1000
            self.latency_history.append(latency)
            
            return tone
            
        except Exception as e:
            print(f"Error determining tone: {str(e)}")
            return 'neutral'

    def determine_tone(self, emotion: str, emotion_conf: float, sentiment: float, sentiment_conf: float) -> str:
        """Determine tone with streaming optimization"""
        try:
            # Add to processing queue
            self.input_queue.put((emotion, emotion_conf, sentiment, sentiment_conf))
            
            # Get result from queue with timeout
            try:
                tone = self.result_queue.get(timeout=0.1)  # 100ms timeout
                
                # Apply smoothing
                self.recent_predictions.append(tone)
                if len(self.recent_predictions) == self.smoothing_window:
                    # Get most common tone
                    tone_counts = {}
                    for t in self.recent_predictions:
                        tone_counts[t] = tone_counts.get(t, 0) + 1
                    tone = max(tone_counts.items(), key=lambda x: x[1])[0]
                
                self.current_tone = tone
                return tone
                
            except queue.Empty:
                print("Warning: Processing timeout")
                return self.current_tone
                
        except Exception as e:
            print(f"Error in determine_tone: {str(e)}")
            return self.current_tone

    def get_average_latency(self) -> float:
        """Get average latency from recent history"""
        if not self.latency_history:
            return 0.0
        return sum(self.latency_history) / len(self.latency_history)

    def adjust_tone(self, text: str, emotion: str, sentiment: str) -> bytes:
        """Generate speech with the current tone settings"""
        # Only require API key here
        api_key = os.getenv('ELEVEN_LABS_API_KEY')
        if not api_key:
            raise ValueError("ELEVEN_LABS_API_KEY not found in environment variables")
        start_time = time.time()
        
        # Combine emotion and sentiment for better tone adjustment
        if emotion == 'happy' and sentiment == 'positive':
            settings = self.tone_mappings['happy']
        elif emotion == 'sad' and sentiment == 'negative':
            settings = self.tone_mappings['angry']
        elif emotion == 'angry' and sentiment == 'negative':
            settings = self.tone_mappings['angry']
        else:
            settings = self.tone_mappings['neutral']

        try:
            audio = ElevenLabs(api_key=api_key).generate(
                text=text,
                voice=Voice(
                    voice_id="EXAVITQu4vr4xnSDxMaL",
                    settings=VoiceSettings(
                        stability=settings['stability'],
                        similarity_boost=settings['similarity_boost']
                    )
                ),
                model="eleven_monolingual_v1"
            )
            
            # Calculate latency
            latency = (time.time() - start_time) * 1000  # Convert to milliseconds
            self.latency_history.append(latency)
            
            return audio
            
        except Exception as e:
            print(f"Error generating audio: {str(e)}")
            return None
    
    def _emotion_to_value(self, emotion: str) -> float:
        """Convert emotion to a numerical value"""
        emotion_values = {
            "happy": 1.0,
            "neutral": 0.0,
            "sad": -0.5,
            "angry": -1.0
        }
        return emotion_values.get(emotion, 0.0)
    
    def get_voice_settings(self) -> Dict:
        """Get the current voice settings based on tone"""
        return self.tone_mappings[self.current_tone]
    
    def generate_speech(self, text: str) -> bytes:
        """Generate speech with the current tone settings"""
        settings = self.get_voice_settings()
        audio = ElevenLabs(api_key=os.getenv('ELEVEN_LABS_API_KEY')).generate(
            text=text,
            voice=Voice(
                voice_id="EXAVITQu4vr4xnSDxMaL",
                settings=VoiceSettings(
                    stability=0.5,
                    similarity_boost=0.5
                )
            ),
            model="eleven_monolingual_v1"
        )
        return audio 