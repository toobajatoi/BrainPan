import sys
from emotion_detector import EmotionDetector
from sentiment_checker import SentimentChecker
from tone_switcher import ToneSwitcher
from stt_engine import STTEngine

import numpy as np

# Usage: python demo_pipeline.py input.wav

def main(wav_path):
    emotion_detector = EmotionDetector()
    sentiment_checker = SentimentChecker()
    tone_switcher = ToneSwitcher()
    stt_engine = STTEngine()

    # 1. Emotion detection (1s chunks)
    emotion_results = emotion_detector.process_wav_file(wav_path)
    print("\nEmotion Detection Results:")
    for ts, emotion, conf in emotion_results:
        print(f"{ts:.1f}s: {emotion} (conf: {conf:.2f})")

    # 2. STT + Sentiment (2s chunks)
    stt_results = stt_engine.transcribe_wav(wav_path, chunk_duration=2.0)
    sentiment_results = []
    print("\nSentiment Analysis Results:")
    for ts, transcript in stt_results:
        if transcript.strip():
            sentiment, conf = sentiment_checker.analyze_sentiment(transcript)
            sentiment_results.append((ts, transcript, sentiment, conf))
            print(f"{ts:.1f}s: '{transcript}' -> sentiment: {sentiment:.2f} (conf: {conf:.2f})")
        else:
            sentiment_results.append((ts, transcript, 0.0, 0.0))

    # 3. Combine and generate TTS for each 2s segment
    print("\nTone & TTS Synthesis:")
    tts_audio = b''
    for i, (ts, transcript, sentiment, sent_conf) in enumerate(sentiment_results):
        # Find the closest emotion result for this timestamp
        emotion_idx = int(ts)
        if emotion_idx < len(emotion_results):
            _, emotion, emo_conf = emotion_results[emotion_idx]
        else:
            emotion, emo_conf = 'neutral', 1.0
        tone = tone_switcher.determine_tone(emotion, emo_conf, sentiment, sent_conf)
        print(f"{ts:.1f}s: Emotion={emotion} (conf={emo_conf:.2f}), Sentiment={sentiment:.2f} (conf={sent_conf:.2f}) -> Tone={tone}")
        # Generate TTS
        if transcript.strip():
            audio = tone_switcher.adjust_tone(transcript, emotion, sentiment)
            if hasattr(audio, '__iter__') and not isinstance(audio, (bytes, bytearray)):
                audio_bytes = b''.join(audio)
            else:
                audio_bytes = audio
            tts_audio += audio_bytes
    # Save combined TTS audio
    with open("demo_output.wav", "wb") as f:
        f.write(tts_audio)
    print("\nDemo TTS audio saved to 'demo_output.wav'.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python demo_pipeline.py input.wav")
        sys.exit(1)
    main(sys.argv[1]) 