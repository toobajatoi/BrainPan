# Emotion Detection System

A real-time emotion detection system that analyzes audio input to detect emotions (angry, happy, sad, neutral) using acoustic features and machine learning.

## Features

- Real-time audio emotion detection
- Web-based dashboard interface
- Support for multiple emotions (angry, happy, sad, neutral)
- Low-latency processing
- Audio feature extraction and analysis
- Test audio generation for validation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/toobajatoi/BrainPan.git
cd BrainPan
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the emotion detection dashboard:
```bash
python emotion_dashboard.py
```

2. Access the dashboard:
- Local URL: http://127.0.0.1:7860
- The dashboard will also provide a public URL for sharing

3. Generate test audio samples:
```bash
python test_audio/generate_test_audio.py
```

## Project Structure

- `emotion_detector.py`: Core emotion detection logic
- `emotion_dashboard.py`: Web-based dashboard interface
- `test_audio/`: Test audio generation and validation
- `requirements.txt`: Project dependencies

## How It Works

The system uses acoustic features to detect emotions:
- Intensity
- Spectral centroid
- Pitch
- Zero crossing rate

Each emotion has specific thresholds:
- Angry: High intensity, high centroid, high ZCR
- Happy: Moderate intensity, moderate-high centroid, moderate pitch
- Sad: Low intensity, low centroid, low pitch
- Neutral: Moderate values across all features

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Tooba Jatoi 