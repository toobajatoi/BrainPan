import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformer_emotion_detector import TransformerEmotionDetector
import os
from sklearn.model_selection import train_test_split
import logging
from torch.nn.utils.rnn import pad_sequence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAVDESSDataset(Dataset):
    def __init__(self, data_path, labels_df, processor, transform=None):
        self.data_path = data_path
        self.labels_df = labels_df
        self.processor = processor
        self.transform = transform
        
        # Create emotion to index mapping
        self.emotion_map = {
            "neu": 0,  # neutral
            "cal": 1,  # calm
            "hap": 2,  # happy
            "sad": 3,  # sad
            "ang": 4,  # angry
            "fea": 5,  # fear
            "dis": 6,  # disgust
            "sur": 7   # surprise
        }
        
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        audio_path = self.labels_df.iloc[idx]['file']
        emotion = self.labels_df.iloc[idx]['emotion']
        label = self.emotion_map[emotion]
        
        # Load and preprocess audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample to 16kHz if needed
        target_sample_rate = 16000
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            waveform = resampler(waveform)
            sample_rate = target_sample_rate
        
        # Normalize audio
        waveform = waveform / torch.max(torch.abs(waveform))
        
        # Process audio for model input
        inputs = self.processor(
            waveform.squeeze().numpy(),
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True
        )
        
        return inputs.input_values.squeeze(), label

def collate_fn(batch):
    audios, labels = zip(*batch)
    audios = [torch.tensor(a) for a in audios]
    audios_padded = pad_sequence(audios, batch_first=True)
    labels = torch.tensor(labels)
    return audios_padded, labels

def main():
    # Initialize emotion detector
    detector = TransformerEmotionDetector()
    
    # Load RAVDESS dataset
    data_path = "data/raw"
    labels_df = pd.read_csv("data/processed/train.csv")
    
    # Split data into train and validation sets
    train_df, val_df = train_test_split(labels_df, test_size=0.2, random_state=42)
    
    # Create datasets
    train_dataset = RAVDESSDataset(data_path, train_df, detector.processor)
    val_dataset = RAVDESSDataset(data_path, val_df, detector.processor)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate_fn)
    
    # Train the model
    logger.info("Starting training...")
    detector.train(train_loader, num_epochs=10)
    
    # Save the trained model
    os.makedirs("models", exist_ok=True)
    detector.save_model("models/emotion_detector.pt")
    
    # Evaluate on validation set
    detector.model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for audio_data, labels in val_loader:
            audio_data = audio_data.to(detector.device)
            labels = labels.to(detector.device)
            
            outputs = detector.model(audio_data)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    logger.info(f"Validation accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main() 