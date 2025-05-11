import torch
import torch.nn as nn
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import numpy as np
import time
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionClassifier(nn.Module):
    def __init__(self, num_emotions=8):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_emotions)
        )
        
    def forward(self, x):
        outputs = self.wav2vec2(x)
        hidden_states = outputs.last_hidden_state
        pooled = torch.mean(hidden_states, dim=1)
        return self.classifier(pooled)

class TransformerEmotionDetector:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = EmotionClassifier().to(self.device)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"Loaded model from {model_path}")
        
        self.model.eval()
        
        self.emotion_map = {
            0: "neu",  # neutral
            1: "cal",  # calm
            2: "hap",  # happy
            3: "sad",  # sad
            4: "ang",  # angry
            5: "fea",  # fear
            6: "dis",  # disgust
            7: "sur"   # surprise
        }
        
        self.sample_rate = 16000  # Wav2Vec2 expects 16kHz audio
        
    def preprocess_audio(self, audio_data, sample_rate):
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
            
        # Resample if necessary
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
            audio_data = resampler(torch.from_numpy(audio_data))
            audio_data = audio_data.numpy()
            
        # Normalize audio
        audio_data = audio_data / np.max(np.abs(audio_data))
        
        return audio_data
    
    def process_audio(self, audio_data, sample_rate):
        start_time = time.time()
        
        try:
            # Preprocess audio
            audio_data = self.preprocess_audio(audio_data, sample_rate)
            
            # Process audio for model input
            inputs = self.processor(
                audio_data,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True
            )
            
            # Move to device and get predictions
            inputs = inputs.input_values.to(self.device)
            with torch.no_grad():
                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_emotion = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_emotion].item()
                
            emotion = self.emotion_map[predicted_emotion]
            latency = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            logger.info(f"Detected emotion: {emotion} with confidence: {confidence:.2f}")
            logger.info(f"Processing latency: {latency:.2f}ms")
            
            return {
                "emotion": emotion,
                "confidence": confidence,
                "latency": latency
            }
            
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            return {
                "emotion": "neu",
                "confidence": 0.0,
                "latency": (time.time() - start_time) * 1000
            }
    
    def train(self, train_loader, num_epochs=10, learning_rate=1e-4):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for audio_data, labels in train_loader:
                audio_data = audio_data.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(audio_data)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
            accuracy = 100 * correct / total
            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
            
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")
        
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        logger.info(f"Model loaded from {path}") 