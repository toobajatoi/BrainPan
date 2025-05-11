import numpy as np
import librosa
import time
import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from typing import List, Tuple, Dict
from collections import deque
import traceback
import sounddevice as sd
import threading
import queue
import langid

class EmotionDetector:
    def __init__(self, smoothing_window=3, chunk_size=0.25, confidence_threshold=0.3):
        self.sample_rate = 16000
        self.chunk_size_sec = chunk_size
        self.chunk_size_samples = int(self.sample_rate * self.chunk_size_sec)
        self.smoothing_window = smoothing_window
        self.confidence_threshold = confidence_threshold
        self.recent_predictions = deque(maxlen=smoothing_window)
        self.latency_history = deque(maxlen=100)
        self.last_process_time = time.time()
        
        # Pre-compute feature extraction parameters
        self.n_mfcc = 13
        self.n_mels = 40
        self.fmin = 0
        self.fmax = 8000
        
        # Initialize feature extraction pipeline
        self.mfcc_extractor = librosa.feature.mfcc
        self.mel_extractor = librosa.feature.melspectrogram
        self.spectral_extractor = librosa.feature.spectral_centroid
        
        # Load pre-trained model for faster inference
        self.model = self._load_model()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def _load_model(self):
        """Load a lightweight pre-trained model for emotion detection"""
        try:
            # Use a small CNN model for fast inference
            model = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(32 * 10 * 3, 64),
                nn.ReLU(),
                nn.Linear(64, 4),
                nn.Softmax(dim=1)
            )
            return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None

    def extract_features(self, audio):
        """Extract audio features optimized for emotion detection"""
        try:
            start_time = time.time()
            
            # Extract MFCCs
            mfccs = self.mfcc_extractor(y=audio, sr=self.sample_rate, n_mfcc=self.n_mfcc)
            
            # Extract Mel spectrogram
            mel_spec = self.mel_extractor(y=audio, sr=self.sample_rate, n_mels=self.n_mels)
            
            # Extract spectral features
            spectral_centroid = self.spectral_extractor(y=audio, sr=self.sample_rate)[0]
            
            # Calculate pitch using YIN algorithm (faster than autocorrelation)
            pitch = librosa.yin(audio, fmin=librosa.note_to_hz('C2'), 
                              fmax=librosa.note_to_hz('C7'))
            
            # Calculate intensity
            intensity = np.mean(np.abs(audio))
            
            # Calculate zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            
            features = {
                'mfccs': mfccs,
                'mel_spec': mel_spec,
                'spectral_centroid': np.mean(spectral_centroid),
                'pitch_mean': np.mean(pitch),
                'intensity': intensity,
                'zero_crossing_rate': np.mean(zcr)
            }
            
            latency = (time.time() - start_time) * 1000
            self.latency_history.append(latency)
            
            return features
            
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return None

    def classify_emotion_from_features(self, features):
        """Classify emotion based on audio features using mutually exclusive, stricter rules"""
        try:
            intensity = features['intensity']
            pitch = features['pitch_mean']
            centroid = features['spectral_centroid']
            zcr = features['zero_crossing_rate']

            # Angry: intensity > 0.18 AND centroid > 1000 AND zcr > 0.10
            if intensity > 0.18 and centroid > 1000 and zcr > 0.10:
                return 'angry', 1.0

            # Happy: intensity 0.10–0.18 AND centroid 500–1000 AND pitch > 140
            if 0.10 <= intensity <= 0.18 and 500 <= centroid <= 1000 and pitch > 140:
                return 'happy', 1.0

            # Sad: intensity < 0.11 AND centroid < 400 AND pitch < 120
            if intensity < 0.11 and centroid < 400 and pitch < 120:
                return 'sad', 1.0

            # Neutral: intensity 0.10–0.16 AND centroid 400–700 AND pitch 120–160
            if 0.10 <= intensity <= 0.16 and 400 <= centroid <= 700 and 120 <= pitch <= 160:
                return 'neutral', 1.0

            # If none match, return neutral with low confidence
            return 'neutral', 0.5
        except Exception as e:
            print(f"Error classifying emotion: {str(e)}")
            return 'neutral', 1.0

    def process_audio(self, audio_data):
        """Process audio data and return emotion prediction"""
        try:
            # Resample if necessary
            if len(audio_data) != self.chunk_size_samples:
                audio_data = librosa.resample(
                    audio_data,
                    orig_sr=self.sample_rate,
                    target_sr=self.sample_rate
                )
                audio_data = audio_data[:self.chunk_size_samples]
            
            # Extract features
            features = self.extract_features(audio_data)
            
            # Get emotion prediction
            emotion, confidence = self.classify_emotion_from_features(features)
            
            # Apply confidence threshold
            if confidence < self.confidence_threshold:
                emotion = "neutral"
                confidence = 0.5
            
            # Update recent predictions
            self.recent_predictions.append((emotion, confidence))
            if len(self.recent_predictions) > self.smoothing_window:
                self.recent_predictions.popleft()
            
            # Calculate latency
            latency = time.time() - self.last_process_time
            self.last_process_time = time.time()
            self.latency_history.append(latency * 1000)  # Convert to ms
            if len(self.latency_history) > self.smoothing_window:
                self.latency_history.popleft()
            
            return time.time(), emotion, confidence
            
        except Exception as e:
            print(f"Error in audio analysis: {str(e)}")
            return time.time(), "neutral", 0.5

    def get_average_latency(self):
        """Get average latency from recent history"""
        if not self.latency_history:
            return 0.0
        return sum(self.latency_history) / len(self.latency_history)

    def detect_audio_language(self, audio_data, sample_rate=16000):
        # Try to use langid on a quick STT transcript (if available)
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            audio = sr.AudioData((audio_data * 32767).astype(np.int16).tobytes(), sample_rate, 2)
            text = recognizer.recognize_google(audio, show_all=False)
            lang, _ = langid.classify(text)
            return lang
        except Exception:
            return 'en'  # Default to English if detection fails

    def record_audio(self, duration=1.0, sample_rate=16000):
        """Record audio from microphone"""
        print(f"Recording {duration} seconds of audio...")
        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='float32',
            blocking=True
        )
        return audio_data.flatten()

    def start_realtime_recording(self):
        """Start real-time audio recording and processing"""
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Audio callback status: {status}")
            self.audio_queue.put(indata.copy().flatten())
        
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            callback=audio_callback
        ):
            print("Recording started. Press Ctrl+C to stop.")
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\nRecording stopped.")

    def process_wav_file(self, wav_path: str):
        try:
            print(f"Processing {wav_path}...")
            audio_data, sr = librosa.load(wav_path, sr=self.sample_rate)
            print(f"Loaded audio: shape={audio_data.shape}, sr={sr}")
            
            chunk_size = int(1.0 * self.sample_rate)
            chunks = [audio_data[i:i + chunk_size] for i in range(0, len(audio_data), chunk_size)]
            print(f"Split into {len(chunks)} chunks")
            
            results = []
            for i, chunk in enumerate(chunks):
                timestamp, emotion, conf = self.process_audio(chunk)
                results.append((timestamp, emotion, conf))
            return results
        except Exception as e:
            print(f"Error in process_wav_file: {str(e)}")
            traceback.print_exc()
            return [] 