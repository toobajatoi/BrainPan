import os
import requests
import zipfile
import shutil
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def download_file(url, filename):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(filename, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()

def prepare_ravdess_data():
    """Download and prepare RAVDESS dataset."""
    # Create directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('test_audio/real', exist_ok=True)
    
    # Download RAVDESS dataset
    print("Downloading RAVDESS dataset...")
    url = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
    zip_path = "data/raw/ravdess.zip"
    
    if not os.path.exists(zip_path):
        download_file(url, zip_path)
    
    # Extract the dataset
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("data/raw")
    
    # Process audio files
    print("Processing audio files...")
    audio_files = []
    base_path = "data/raw"
    
    if not os.path.exists(base_path):
        print(f"Error: {base_path} does not exist!")
        return
    
    print(f"Searching for WAV files in {base_path}...")
    for root, dirs, files in os.walk(base_path):
        print(f"Checking directory: {root}")
        print(f"Found {len(files)} files")
        for file in files:
            if file.endswith(".wav"):
                print(f"Processing {file}")
                # RAVDESS filename format: modality-vocal_channel-emotion-intensity-statement-repetition-actor.wav
                parts = file.split('-')
                if len(parts) >= 7:
                    emotion_code = parts[2]
                    intensity = parts[3]
                    statement = parts[4]
                    actor = parts[6].split('.')[0]
                    
                    # Map RAVDESS emotions to our categories
                    emotion_map = {
                        '01': 'neu',  # neutral
                        '02': 'cal',  # calm
                        '03': 'hap',  # happy
                        '04': 'sad',  # sad
                        '05': 'ang',  # angry
                        '06': 'fea',  # fearful
                        '07': 'dis',  # disgust
                        '08': 'sur'   # surprised
                    }
                    
                    if emotion_code in emotion_map:
                        audio_files.append({
                            'file': os.path.join(root, file),
                            'emotion': emotion_map[emotion_code],
                            'intensity': intensity,
                            'statement': statement,
                            'actor': actor
                        })
                        print(f"Added {file} with emotion {emotion_map[emotion_code]}")
    
    # Convert to DataFrame
    df = pd.DataFrame(audio_files)
    print(f"\nFound {len(df)} audio files")
    
    if len(df) == 0:
        print("Error: No audio files found!")
        return
    
    print("Emotion distribution:")
    print(df['emotion'].value_counts())
    
    # Split into train/test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['emotion'])
    
    # Save test files
    print("\nSaving test files...")
    test_files = {
        'happy': test_df[test_df['emotion'] == 'hap'].iloc[0],
        'sad': test_df[test_df['emotion'] == 'sad'].iloc[0],
        'angry': test_df[test_df['emotion'] == 'ang'].iloc[0],
        'neutral': test_df[test_df['emotion'] == 'neu'].iloc[0]
    }
    
    for emotion, row in test_files.items():
        src = row['file']
        dst = f"test_audio/real/{emotion}.wav"
        shutil.copy2(src, dst)
        print(f"Copied {src} to {dst}")
    
    # Update labels.csv
    print("\nUpdating labels.csv...")
    with open('test_audio/labels.csv', 'w') as f:
        f.write("happy.wav,hap,positive,I'm so happy to talk to you!\n")
        f.write("sad.wav,sad,negative,I'm feeling really down today.\n")
        f.write("angry.wav,ang,negative,Why did this happen again?!\n")
        f.write("neutral.wav,neu,neutral,Okay, I understand.\n")
    
    # Save processed data
    print("\nSaving processed data...")
    train_df.to_csv('data/processed/train.csv', index=False)
    test_df.to_csv('data/processed/test.csv', index=False)
    
    print("\nDone! Real emotional audio samples are ready in test_audio/real/")
    print("Training data is available in data/processed/train.csv")
    print("Test data is available in data/processed/test.csv")

if __name__ == "__main__":
    prepare_ravdess_data() 