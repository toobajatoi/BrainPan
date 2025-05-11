import gradio as gr
import numpy as np
from emotion_detector import EmotionDetector
import time
from transformers import pipeline
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import librosa
from langdetect import detect
from sentiment_checker import SentimentChecker
from tone_switcher import ToneSwitcher
import sounddevice as sd
import threading
import queue
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import io

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
detector = EmotionDetector(smoothing_window=5, chunk_size=0.25, confidence_threshold=0.5)
sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

# Emotion colors
EMOTION_COLORS = {
    'neu': '#808080',  # Gray
    'hap': '#FFD700',  # Gold
    'ang': '#FF4500',  # Red
    'sad': '#4169E1'   # Blue
}

class EmotionDashboard:
    def __init__(self):
        self.detector = EmotionDetector()
        self.checker = SentimentChecker()
        self.switcher = ToneSwitcher()
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.recording_thread = None
        
        # Initialize performance metrics
        self.emotion_history = []
        self.sentiment_history = []
        self.tone_history = []
        self.latency_history = []
        
    def start_recording(self):
        """Start recording audio"""
        self.is_recording = True
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.recording_thread.start()
        
    def stop_recording(self):
        """Stop recording audio"""
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join()
            
    def _record_audio(self):
        """Record audio in a background thread"""
        with sd.InputStream(
            samplerate=self.detector.sample_rate,
            channels=1,
            dtype='float32'
        ) as stream:
            while self.is_recording:
                audio_data, _ = stream.read(int(self.detector.sample_rate * 0.25))  # 250ms chunks
                self.audio_queue.put(audio_data.flatten())
                
    def process_audio(self, audio_data):
        """Process audio data and return results"""
        try:
            # 1. Emotion Detection
            start_time = time.time()
            timestamp, emotion, emotion_conf = self.detector.process_audio(audio_data)
            emotion_latency = (time.time() - start_time) * 1000
            
            # 2. Sentiment Analysis (using a sample text for now)
            start_time = time.time()
            sentiment, sentiment_score, sentiment_conf = self.checker.analyze_sentiment(
                "This is a test sentence for sentiment analysis."
            )
            sentiment_latency = (time.time() - start_time) * 1000
            
            # 3. Tone Switching
            start_time = time.time()
            tone = self.switcher.determine_tone(
                emotion, emotion_conf,
                sentiment_score, sentiment_conf
            )
            tone_latency = (time.time() - start_time) * 1000
            
            # Update history
            self.emotion_history.append((emotion, emotion_conf))
            self.sentiment_history.append((sentiment, sentiment_score))
            self.tone_history.append(tone)
            self.latency_history.append({
                'emotion': emotion_latency,
                'sentiment': sentiment_latency,
                'tone': tone_latency,
                'total': emotion_latency + sentiment_latency + tone_latency
            })
            
            # Create visualization
            fig = self._create_visualization()
            
            return {
                'emotion': f"{emotion} ({emotion_conf:.2f})",
                'sentiment': f"{sentiment} ({sentiment_score:.2f})",
                'tone': tone,
                'latency': f"Total: {emotion_latency + sentiment_latency + tone_latency:.2f} ms",
                'plot': fig
            }
            
        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            return {
                'emotion': "Error",
                'sentiment': "Error",
                'tone': "Error",
                'latency': "Error",
                'plot': None
            }
            
    def _create_visualization(self):
        """Create a visualization of the results"""
        fig = Figure(figsize=(10, 6))
        
        # Plot emotion confidence over time
        ax1 = fig.add_subplot(311)
        emotions = [e[0] for e in self.emotion_history[-20:]]
        confidences = [e[1] for e in self.emotion_history[-20:]]
        ax1.plot(confidences)
        ax1.set_title('Emotion Confidence')
        ax1.set_ylim(0, 1)
        
        # Plot sentiment score over time
        ax2 = fig.add_subplot(312)
        sentiments = [s[1] for s in self.sentiment_history[-20:]]
        ax2.plot(sentiments)
        ax2.set_title('Sentiment Score')
        ax2.set_ylim(-1, 1)
        
        # Plot latency over time
        ax3 = fig.add_subplot(313)
        latencies = [l['total'] for l in self.latency_history[-20:]]
        ax3.plot(latencies)
        ax3.set_title('Total Latency (ms)')
        ax3.set_ylim(0, 200)
        
        fig.tight_layout()
        
        # Convert to image
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        return buf
        
    def create_interface(self):
        """Create the Gradio interface"""
        with gr.Blocks(title="Emotion Detection Dashboard") as interface:
            gr.Markdown("# Real-time Emotion Detection Dashboard")
            
            with gr.Row():
                with gr.Column():
                    start_btn = gr.Button("Start Recording")
                    stop_btn = gr.Button("Stop Recording")
                    
            with gr.Row():
                with gr.Column():
                    emotion_output = gr.Textbox(label="Emotion")
                    sentiment_output = gr.Textbox(label="Sentiment")
                    tone_output = gr.Textbox(label="Tone")
                    latency_output = gr.Textbox(label="Latency")
                    
            with gr.Row():
                plot_output = gr.Image(label="Performance Visualization")
                
            def start_recording():
                self.start_recording()
                return "Recording started..."
                
            def stop_recording():
                self.stop_recording()
                return "Recording stopped..."
                
            def process_audio():
                if not self.audio_queue.empty():
                    audio_data = self.audio_queue.get()
                    return self.process_audio(audio_data)
                return None
                
            start_btn.click(start_recording)
            stop_btn.click(stop_recording)
            
            # Set up periodic processing
            interface.load(process_audio, every=0.25)  # Process every 250ms
            
        return interface

def create_emotion_chart(emotions, confidences):
    """Create a bar chart for emotion probabilities"""
    fig = go.Figure(data=[
        go.Bar(
            x=list(emotions.keys()),
            y=list(emotions.values()),
            marker_color=[EMOTION_COLORS.get(emo, '#808080') for emo in emotions.keys()]
        )
    ])
    fig.update_layout(
        title="Emotion Probabilities",
        xaxis_title="Emotion",
        yaxis_title="Probability",
        yaxis_range=[0, 1],
        showlegend=False
    )
    return fig

def create_sentiment_gauge(score):
    """Create a gauge chart for sentiment score"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [-1, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [-1, -0.5], 'color': "red"},
                {'range': [-0.5, 0], 'color': "orange"},
                {'range': [0, 0.5], 'color': "lightgreen"},
                {'range': [0.5, 1], 'color': "green"}
            ]
        },
        title={'text': "Sentiment Score"}
    ))
    return fig

def get_tone(emotion, sentiment_score):
    """Determine the appropriate tone based on emotion and sentiment"""
    if emotion == 'hap' and sentiment_score > 0:
        return "bright & upbeat", "#FFD700"  # Gold
    elif emotion in ['ang', 'sad'] and sentiment_score < 0:
        return "calm & slow", "#4169E1"  # Blue
    elif emotion == 'hap':
        return "bright & upbeat", "#FFD700"  # Gold
    elif emotion == 'ang':
        return "calm & slow", "#4169E1"  # Blue
    elif emotion == 'sad':
        return "gentle & soothing", "#4169E1"  # Blue
    else:
        return "neutral", "#808080"  # Gray

def analyze_text(text):
    """Analyze sentiment of text input"""
    if not text:
        return "No text", 0.0, "default", "#808080", None, None, 0.0
    start_time = time.time()
    try:
        if not hasattr(analyze_text, 'checker'):
            analyze_text.checker = SentimentChecker()
        checker = analyze_text.checker
        sentiment, score, confidence = checker.analyze_sentiment(text)
        sentiment_label = sentiment
        sentiment_score = score
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        sentiment_label = "error"
        sentiment_score = 0.0
        confidence = 0.0
    latency = checker.get_average_latency() if 'checker' in locals() else (time.time() - start_time) * 1000
    logger.info(f"Text Analysis - Sentiment: {sentiment_label}, Score: {sentiment_score:.2f}, Latency: {latency:.2f}ms")
    tone, tone_color = get_tone('neu', sentiment_score)
    sentiment_gauge = create_sentiment_gauge(sentiment_score)
    return sentiment_label, sentiment_score, tone, tone_color, sentiment_gauge, None, latency

def analyze_audio(audio):
    """Analyze audio input for emotion and sentiment"""
    try:
        if audio is None:
            return "No audio", 0.0, "default", "#808080", None, None, 0.0
        sr, data = audio
        data = data.astype(np.float32) / 32768.0
        if len(data.shape) > 1:
            data = data[:, 0]
        if sr != 16000:
            data = librosa.resample(data, orig_sr=sr, target_sr=16000)
            sr = 16000
        # Get emotion detection (returns: ts, emotion, conf)
        ts, emotion, confidence = detector.process_audio(data)
        # Get sentiment analysis of the detected emotion label (for visualization)
        sentiment = sentiment_analyzer(emotion)[0]['label'].lower()
        score = sentiment_analyzer(emotion)[0]['score']
        if sentiment == 'negative':
            score = -score
        tone, tone_color = get_tone(emotion, score)
        emotions = {'neu': 0.0, 'hap': 0.0, 'ang': 0.0, 'sad': 0.0}
        emotions[emotion[:3]] = confidence if emotion[:3] in emotions else 0.0
        emotion_chart = create_emotion_chart(emotions, [confidence])
        sentiment_gauge = create_sentiment_gauge(score)
        # Estimate latency as 0 for now (or you can time the call above)
        audio_latency = 0.0
        return emotion, confidence, tone, tone_color, [emotion_chart, sentiment_gauge], None, audio_latency
    except Exception as e:
        print(f"Error in audio analysis: {str(e)}")
        return "Error", 0.0, "Error", "#808080", None, None, 0.0

def process_input(audio, text):
    """Process both audio and text inputs"""
    results = []
    charts = []
    if audio is not None:
        emotion, conf, tone, tone_color, emotion_charts, lang, audio_latency = analyze_audio(audio)
        results.append(f"Audio Analysis:\nEmotion: {emotion}\nConfidence: {conf:.2f}\nLanguage: {lang}\nTone: {tone}\nLatency: {audio_latency:.2f}ms")
        if emotion_charts:
            charts.extend(emotion_charts)
    if text:
        sentiment, score, tone, tone_color, sentiment_gauge, lang, text_latency = analyze_text(text)
        results.append(f"Text Analysis:\nSentiment: {sentiment}\nScore: {score:.2f}\nLanguage: {lang}\nTone: {tone}\nLatency: {text_latency:.2f}ms")
        charts.append(sentiment_gauge)
    # Combine all charts into a single figure if there are multiple
    if len(charts) > 1:
        specs = []
        for chart in charts:
            if hasattr(chart, 'data') and len(chart.data) > 0 and getattr(chart.data[0], 'type', None) == 'indicator':
                specs.append([{"type": "indicator"}])
            else:
                specs.append([{"type": "xy"}])
        fig = make_subplots(rows=len(charts), cols=1, subplot_titles=[f"Chart {i+1}" for i in range(len(charts))], specs=specs)
        for i, chart in enumerate(charts):
            for trace in chart.data:
                fig.add_trace(trace, row=i+1, col=1)
        fig.update_layout(height=300*len(charts), showlegend=False)
        return ("\n\n".join(results), fig)
    elif len(charts) == 1:
        return ("\n\n".join(results), charts[0])
    else:
        return ("\n\n".join(results), None)

# Custom CSS for better visualization
css = """
.gradio-container {
    max-width: 1200px;
    margin: 0 auto;
}
.container {
    display: flex;
    flex-direction: column;
    gap: 20px;
}
.chart-container {
    display: flex;
    justify-content: space-around;
    flex-wrap: wrap;
    gap: 20px;
}
"""

# Create the interface
iface = gr.Interface(
    fn=process_input,
    inputs=[
        gr.Audio(sources=["microphone"], type="numpy", label="Speak to test emotion"),
        gr.Textbox(label="Enter text for sentiment analysis", placeholder="Type something...")
    ],
    outputs=[
        gr.Textbox(label="Analysis Results", lines=10),
        gr.Plot(label="Visualizations")
    ],
    live=False,
    title="Real-Time Emotion & Sentiment Analysis",
    description="""This dashboard provides:
    1. Real-time emotion detection from audio
    2. Sentiment analysis from text
    3. Automatic tone switching based on emotion and sentiment
    4. Performance metrics and logging
    5. Visual analysis of results""",
    css=css
)

if __name__ == "__main__":
    iface.launch(share=True)  # Enable sharing for external access 