import numpy as np
import librosa
import time
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from typing import List, Tuple, Dict
from collections import deque
import traceback
import sounddevice as sd
import threading
import queue

class EmotionDetector:
    def __init__(self, smoothing_window=3, chunk_size=0.25):
        # Load public, lightweight model for English
        self.model_name = "superb/wav2vec2-base-superb-er"
        
        # Initialize model
        print("Loading emotion detection model...")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
        self.model = AutoModelForAudioClassification.from_pretrained(self.model_name)
        
        # Move model to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Get emotion labels
        self.id2label = self.model.config.id2label
        print(f"Model emotion labels: {self.id2label}")
        
        # Initialize other attributes
        self.smoothing_window = smoothing_window
        self.recent_predictions = []
        self.latency_history = []
        self.sample_rate = 16000
        self.chunk_size = chunk_size
        
        # Real-time processing setup
        self.audio_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        self.is_processing = False
        self.processing_thread = None
        
        # Warm-up call to ensure first inference is fast
        print("Warming up emotion model...")
        dummy_audio = np.zeros(int(self.chunk_size * self.sample_rate), dtype=np.float32)
        self.process_audio(dummy_audio)
        print("Warm-up complete.")
        
        # Start processing thread
        self.start_processing()

    def start_processing(self):
        """Start the background processing thread"""
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._process_audio_stream)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def stop_processing(self):
        """Stop the background processing thread"""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join()

    def _process_audio_stream(self):
        """Background thread for processing audio chunks"""
        while self.is_processing:
            try:
                if not self.audio_queue.empty():
                    audio_data = self.audio_queue.get()
                    result = self.process_audio(audio_data)
                    self.result_queue.put(result)
            except Exception as e:
                print(f"Error in processing thread: {str(e)}")
                traceback.print_exc()
            time.sleep(0.001)  # Small sleep to prevent CPU overload

    def process_audio(self, audio_data, sample_rate=16000):
        try:
            start_time = time.time()
            
            # Use only chunk_size seconds of audio
            chunk_len = int(self.chunk_size * sample_rate)
            if len(audio_data) > chunk_len:
                audio_data = audio_data[:chunk_len]
            else:
                audio_data = np.pad(audio_data, (0, chunk_len - len(audio_data)))
                
            # Ensure audio data is in the correct format
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.array(audio_data)
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Prepare input
            inputs = self.feature_extractor(
                audio_data, 
                sampling_rate=sample_rate, 
                return_tensors="pt"
            ).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
                
            pred_idx = torch.argmax(probs).item()
            emotion_label = self.id2label[pred_idx]
            confidence = probs[pred_idx].item()
            
            # Smoothing
            self.recent_predictions.append((emotion_label, confidence))
            if len(self.recent_predictions) > self.smoothing_window:
                self.recent_predictions.pop(0)
                
            # Latency
            latency = (time.time() - start_time) * 1000
            self.latency_history.append(latency)
            
            return emotion_label, confidence, 'en'
            
        except Exception as e:
            print(f"Error in process_audio: {str(e)}")
            print(f"Audio data shape: {audio_data.shape}, dtype: {audio_data.dtype}")
            traceback.print_exc()
            return 'neutral', 0.0, 'en'

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

    def get_average_latency(self):
        if not self.latency_history:
            return 0.0
        return sum(self.latency_history) / len(self.latency_history)

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
                emotion, conf, lang = self.process_audio(chunk)
                results.append((i, emotion, conf, lang))
            return results
        except Exception as e:
            print(f"Error in process_wav_file: {str(e)}")
            traceback.print_exc()
            return [] 