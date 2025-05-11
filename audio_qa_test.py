import os
import numpy as np
import librosa
# from fast_emotion_detector import FastEmotionDetector
from emotion_detector import EmotionDetector  # Use transformer-based detector
from sentiment_checker import SentimentChecker
from tone_switcher import ToneSwitcher
from tts_engine import TTSEngine
from tabulate import tabulate

# Folder with test audio files and a CSV with ground truth
AUDIO_DIR = 'test_audio'
LABELS_FILE = os.path.join(AUDIO_DIR, 'labels.csv')

# labels.csv format: filename,true_emotion,true_sentiment,text

def load_labels(labels_file):
    labels = []
    with open(labels_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                parts = line.strip().split(',')
                if len(parts) >= 4:
                    labels.append({
                        'filename': parts[0],
                        'true_emotion': parts[1],
                        'true_sentiment': parts[2],
                        'text': ','.join(parts[3:])
                    })
    return labels

def main():
    detector = EmotionDetector()  # Use transformer-based detector
    checker = SentimentChecker()
    switcher = ToneSwitcher()
    tts = TTSEngine()
    labels = load_labels(LABELS_FILE)
    results = []
    for entry in labels:
        wav_path = os.path.join(AUDIO_DIR, entry['filename'])
        print(f"\nProcessing {wav_path}...")
        audio, sr = librosa.load(wav_path, sr=detector.sample_rate)
        timestamp, emotion, emo_conf = detector.process_audio(audio)
        sentiment_result, sent_conf, _, _, _ = checker.analyze_sentiment(entry['text'])
        tone = switcher.determine_tone(emotion, emo_conf, sentiment_result['score'], sent_conf)
        tts.synthesize(entry['text'], tone, play_audio=False)  # Synthesize but don't play
        emotion_correct = emotion == entry['true_emotion']
        sentiment_correct = sentiment_result['sentiment'] == entry['true_sentiment']
        results.append({
            'filename': entry['filename'],
            'emotion': emotion,
            'emo_conf': emo_conf,
            'emotion_correct': emotion_correct,
            'sentiment': sentiment_result['sentiment'],
            'sentiment_correct': sentiment_correct,
            'tone': tone
        })
    # Summary
    emotion_acc = sum(1 for r in results if r['emotion_correct']) / len(results) * 100
    sentiment_acc = sum(1 for r in results if r['sentiment_correct']) / len(results) * 100
    print(f"\nEmotion Detection Accuracy: {emotion_acc:.1f}%")
    print(f"Sentiment Analysis Accuracy: {sentiment_acc:.1f}%")
    headers = ['File', 'Emotion', 'Emo Conf', 'Emo ✓', 'Sentiment', 'Sent ✓', 'Tone']
    table = []
    for r in results:
        table.append([
            r['filename'],
            r['emotion'],
            f"{r['emo_conf']:.2f}",
            '✓' if r['emotion_correct'] else '✗',
            r['sentiment'],
            '✓' if r['sentiment_correct'] else '✗',
            r['tone']
        ])
    print(tabulate(table, headers=headers, tablefmt='grid'))

if __name__ == '__main__':
    main() 