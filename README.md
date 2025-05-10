# BrainPan - Emotion & Sentiment Analysis Dashboard

A real-time emotion and sentiment analysis dashboard that uses machine learning to detect emotions from audio and analyze sentiment from text.

## Features

- Real-time emotion detection from audio input
- Text sentiment analysis
- Interactive visualization of emotions and sentiments
- Confidence scoring for predictions

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/BrainPan.git
cd BrainPan
```

2. Create a virtual environment and activate it:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `config.env` file with your API keys:
```
ELEVEN_LABS_API_KEY=your_api_key_here
```

5. Run the dashboard:
```bash
streamlit run dashboard.py
```

## Project Structure

- `dashboard.py`: Main Streamlit dashboard application
- `emotion_detector.py`: Emotion detection from audio
- `sentiment_checker.py`: Text sentiment analysis
- `tone_switcher.py`: Tone switching based on emotions
- `requirements.txt`: Project dependencies

## Requirements

- Python 3.8+
- Streamlit
- PyTorch
- Transformers
- SoundDevice
- NumPy
- Plotly

## License

MIT License 
MIT License 