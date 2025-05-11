import numpy as np
import soundfile as sf
import librosa
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def generate_test_audio():
    # Create test_audio directory if it doesn't exist
    os.makedirs('test_audio', exist_ok=True)
    
    # Sample rate
    sr = 16000
    
    # Generate 1 second of audio for each emotion
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Angry audio (high intensity, high pitch)
    angry_audio = np.sin(2 * np.pi * 440 * t) * 0.5  # Base tone
    angry_audio += np.sin(2 * np.pi * 880 * t) * 0.3  # Higher harmonics
    angry_audio += np.random.normal(0, 0.1, len(t))  # Noise
    angry_audio = angry_audio * 0.8  # Amplitude
    sf.write('test_audio/angry.wav', angry_audio, sr)
    
    # Happy audio (moderate intensity, high pitch)
    happy_audio = np.sin(2 * np.pi * 523.25 * t) * 0.4  # C5
    happy_audio += np.sin(2 * np.pi * 659.25 * t) * 0.3  # E5
    happy_audio += np.sin(2 * np.pi * 783.99 * t) * 0.2  # G5
    happy_audio = happy_audio * 0.6  # Amplitude
    sf.write('test_audio/happy.wav', happy_audio, sr)
    
    # Sad audio (low intensity, low pitch)
    sad_audio = np.sin(2 * np.pi * 261.63 * t) * 0.3  # C4
    sad_audio += np.sin(2 * np.pi * 329.63 * t) * 0.2  # E4
    sad_audio += np.sin(2 * np.pi * 392.00 * t) * 0.1  # G4
    sad_audio = sad_audio * 0.4  # Amplitude
    sf.write('test_audio/sad.wav', sad_audio, sr)
    
    # Neutral audio (moderate intensity, moderate pitch)
    neutral_audio = np.sin(2 * np.pi * 392.00 * t) * 0.3  # G4
    neutral_audio += np.sin(2 * np.pi * 493.88 * t) * 0.2  # B4
    neutral_audio += np.sin(2 * np.pi * 587.33 * t) * 0.1  # D5
    neutral_audio = neutral_audio * 0.5  # Amplitude
    sf.write('test_audio/neutral.wav', neutral_audio, sr)
    
    print("Generated test audio files in test_audio directory:")
    print("- angry.wav")
    print("- happy.wav")
    print("- sad.wav")
    print("- neutral.wav")

if __name__ == "__main__":
    generate_test_audio()
    from emotion_detector import EmotionDetector
    import librosa
    detector = EmotionDetector()
    for fname in ["angry.wav", "happy.wav", "sad.wav", "neutral.wav"]:
        path = f"test_audio/{fname}"
        audio, sr = librosa.load(path, sr=16000)
        features = detector.extract_features(audio)
        print(f"\nFeatures for {fname}:")
        for k, v in features.items():
            if isinstance(v, np.ndarray):
                print(f"  {k}: mean={np.mean(v):.4f}, std={np.std(v):.4f}")
            else:
                print(f"  {k}: {v:.4f}")
        emotion, confidence = detector.classify_emotion_from_features(features)
        print(f"Predicted emotion: {emotion} (confidence: {confidence:.2f})") 