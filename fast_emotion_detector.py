import numpy as np
import librosa
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import time
import os
import joblib

class FastEmotionDetector:
    def __init__(self, model_path='fast_emotion_model.joblib', sample_rate=16000, chunk_size=1.0):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.model_path = model_path
        self.scaler = StandardScaler()
        self.labels = ['neu', 'hap', 'ang', 'sad']
        if os.path.exists(model_path):
            self.model, self.scaler = joblib.load(model_path)
        else:
            self.model = self._train_dummy_model()
            joblib.dump((self.model, self.scaler), model_path)

    def _extract_features(self, audio_data: np.ndarray) -> np.ndarray:
        mfcc = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        return mfcc_mean

    def _train_dummy_model(self):
        # Generate dummy data for demo (4 classes, 10 samples each)
        X, y = [], []
        for i, label in enumerate(self.labels):
            for _ in range(10):
                dummy_audio = np.random.randn(int(self.chunk_size * self.sample_rate))
                feat = self._extract_features(dummy_audio)
                X.append(feat)
                y.append(i)
        X = np.array(X)
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        model = SVC(probability=True, kernel='linear')
        model.fit(X_scaled, y)
        return model

    def process_audio(self, audio_data: np.ndarray) -> Tuple[float, str, float]:
        start_time = time.time()
        chunk_len = int(self.chunk_size * self.sample_rate)
        if len(audio_data) > chunk_len:
            audio_data = audio_data[:chunk_len]
        else:
            audio_data = np.pad(audio_data, (0, chunk_len - len(audio_data)))
        feat = self._extract_features(audio_data)
        feat_scaled = self.scaler.transform([feat])
        probs = self.model.predict_proba(feat_scaled)[0]
        pred_idx = np.argmax(probs)
        emotion_label = self.labels[pred_idx]
        confidence = probs[pred_idx]
        latency = (time.time() - start_time) * 1000
        print(f"[FastEmotionDetector] Emotion: {emotion_label}, Confidence: {confidence:.2f}, Latency: {latency:.2f} ms")
        return time.time(), emotion_label, confidence 