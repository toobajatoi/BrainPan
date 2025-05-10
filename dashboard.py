import streamlit as st
import time
import numpy as np
import plotly.graph_objects as go
from emotion_detector import EmotionDetector
from sentiment_checker import SentimentChecker
from tone_switcher import ToneSwitcher

def main():
    st.set_page_config(page_title="Emotion & Tone Dashboard", layout="wide")
    st.title("Emotion & Sentiment Analysis")
    
    # Initialize components
    if 'detector' not in st.session_state:
        st.session_state.detector = EmotionDetector()
        st.session_state.checker = SentimentChecker()
        st.session_state.switcher = ToneSwitcher()
        
        # Initialize data storage
        st.session_state.emotion_history = []
        st.session_state.sentiment_history = []
        st.session_state.timestamps = []
    
    # Create columns for input controls
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Audio Input")
        if st.button("🎤 Record Audio (3 seconds)"):
            try:
                with st.spinner("Recording for 3 seconds..."):
                    # Record a single chunk
                    audio = st.session_state.detector.record_audio(duration=3.0)
                    
                    # Process audio
                    emotion, conf, lang = st.session_state.detector.process_audio(audio)
                    st.session_state.emotion_history.append((emotion, conf))
                    st.session_state.timestamps.append(time.time())
                    st.success(f"Detected emotion: {emotion} (confidence: {conf:.2f})")
            except Exception as e:
                st.error(f"Error during recording: {str(e)}")
    
    with col2:
        st.subheader("Text Input")
        text = st.text_input("Enter text for sentiment analysis:", "I am happy today!")
        if st.button("Analyze Sentiment"):
            try:
                sentiment, conf = st.session_state.checker.analyze_sentiment(text)
                st.session_state.sentiment_history.append((sentiment, conf))
                st.success(f"Sentiment: {sentiment:.2f} (confidence: {conf:.2f})")
            except Exception as e:
                st.error(f"Error during sentiment analysis: {str(e)}")
    
    # Display current metrics
    st.subheader("Current Metrics")
    metrics_col1, metrics_col2 = st.columns(2)
    
    with metrics_col1:
        if st.session_state.emotion_history:
            emotion, conf = st.session_state.emotion_history[-1]
            st.metric("Current Emotion", emotion)
            st.metric("Confidence", f"{conf:.2f}")
    
    with metrics_col2:
        if st.session_state.sentiment_history:
            sentiment, conf = st.session_state.sentiment_history[-1]
            st.metric("Current Sentiment", f"{sentiment:.2f}")
            st.metric("Confidence", f"{conf:.2f}")
    
    # Create plots
    st.subheader("History")
    if st.session_state.timestamps:
        # Emotion plot
        fig_emotion = go.Figure()
        emotions = [e[0] for e in st.session_state.emotion_history]
        confidences = [e[1] for e in st.session_state.emotion_history]
        fig_emotion.add_trace(go.Scatter(
            x=st.session_state.timestamps,
            y=emotions,
            mode='lines+markers',
            name='Emotion',
            line=dict(color='blue')
        ))
        fig_emotion.update_layout(
            title='Emotion History',
            xaxis_title='Time',
            yaxis_title='Value',
            showlegend=True
        )
        st.plotly_chart(fig_emotion, use_container_width=True, key=f"emotion_chart_{time.time()}")
        
        # Sentiment plot
        if st.session_state.sentiment_history:
            fig_sentiment = go.Figure()
            sentiments = [s[0] for s in st.session_state.sentiment_history]
            fig_sentiment.add_trace(go.Scatter(
                x=st.session_state.timestamps[-len(sentiments):],
                y=sentiments,
                mode='lines+markers',
                name='Sentiment',
                line=dict(color='green')
            ))
            fig_sentiment.update_layout(
                title='Sentiment History',
                xaxis_title='Time',
                yaxis_title='Value',
                showlegend=True
            )
            st.plotly_chart(fig_sentiment, use_container_width=True, key=f"sentiment_chart_{time.time()}")

if __name__ == "__main__":
    main() 