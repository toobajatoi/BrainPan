from elevenlabs import ElevenLabs, Voice, VoiceSettings
import os
from dotenv import load_dotenv
import soundfile as sf
import numpy as np
from pydub import AudioSegment
import io

def generate_test_audio():
    # Load both .env and config.env for compatibility
    load_dotenv()
    load_dotenv(dotenv_path="config.env")
    api_key = os.getenv('ELEVEN_LABS_API_KEY')
    if not api_key:
        raise ValueError("ELEVEN_LABS_API_KEY not found in environment variables")
    api = ElevenLabs(api_key=api_key)

    # Test cases with different emotions
    test_cases = [
        {
            "file": "test_angry.wav",
            "text": "I am absolutely furious about this situation! This is completely unacceptable!",
            "voice_id": "EXAVITQu4vr4xnSDxMaL",
            "settings": VoiceSettings(stability=0.5, similarity_boost=0.75)
        },
        {
            "file": "test_happy.wav",
            "text": "I'm so excited and happy about this wonderful news! Everything is going great!",
            "voice_id": "EXAVITQu4vr4xnSDxMaL",
            "settings": VoiceSettings(stability=0.5, similarity_boost=0.75)
        },
        {
            "file": "test_sad.wav",
            "text": "I'm feeling really down today. Everything seems so difficult and overwhelming.",
            "voice_id": "EXAVITQu4vr4xnSDxMaL",
            "settings": VoiceSettings(stability=0.5, similarity_boost=0.5)
        },
        {
            "file": "test_neutral.wav",
            "text": "The weather is cloudy today. I'm going to the store to buy some groceries.",
            "voice_id": "EXAVITQu4vr4xnSDxMaL",
            "settings": VoiceSettings(stability=0.5, similarity_boost=0.5)
        },
        {
            "file": "test_mixed.wav",
            "text": "I had a good day at work, but I'm a bit tired now. Looking forward to the weekend though.",
            "voice_id": "EXAVITQu4vr4xnSDxMaL",
            "settings": VoiceSettings(stability=0.5, similarity_boost=0.5)
        }
    ]

    for case in test_cases:
        print(f"Generating {case['file']}...")
        audio = api.generate(
            text=case["text"],
            voice=Voice(
                voice_id=case["voice_id"],
                settings=case["settings"]
            ),
            model="eleven_monolingual_v1"
        )
        # Join generator output if needed
        if hasattr(audio, '__iter__') and not isinstance(audio, (bytes, bytearray)):
            audio_bytes = b''.join(audio)
        else:
            audio_bytes = audio
        # Decode audio bytes using pydub
        audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
        audio_array = np.array(audio_segment.get_array_of_samples(), dtype=np.float32) / 32768.0  # Normalize to [-1, 1]
        # Save as WAV using soundfile
        sf.write(case["file"], audio_array, audio_segment.frame_rate)
        print(f"Saved {case['file']}")

if __name__ == "__main__":
    generate_test_audio() 