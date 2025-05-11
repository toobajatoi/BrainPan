import os
from tts_engine import TTSEngine

test_cases = [
    ("happy.wav", "I'm so happy to talk to you!", "happy"),
    ("sad.wav", "I'm feeling really down today.", "sad"),
    ("angry.wav", "Why did this happen again?!", "angry"),
    ("neutral.wav", "Okay, I understand.", "neutral"),
    ("happy2.wav", "This is wonderful news!", "happy"),
]

def main():
    tts = TTSEngine()
    out_dir = "test_audio"
    os.makedirs(out_dir, exist_ok=True)
    for fname, text, tone in test_cases:
        print(f"Generating {fname} with tone '{tone}'...")
        out_path = os.path.join(out_dir, fname)
        # Synthesize and save using pyttsx3 (local) or ElevenLabs (if available)
        if tts.elevenlabs_api_key:
            audio_data = tts.synthesize(text, tone, play_audio=False)
            if audio_data:
                with open(out_path, 'wb') as f:
                    f.write(audio_data)
        else:
            # Use pyttsx3 to save to file
            tts.engine.save_to_file(text, out_path)
            tts.engine.runAndWait()
    print("All test audio files generated.")

if __name__ == "__main__":
    main() 