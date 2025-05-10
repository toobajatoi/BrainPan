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
        st.session_state.text_emotion_history = []
        st.session_state.timestamps = []
        st.session_state.emotion_latency = []
        st.session_state.sentiment_latency = []
        st.session_state.tone_latency = []
    
    # Create columns for input controls
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Audio Input")
        if st.button("🎤 Record Audio (3 seconds)"):
            try:
                with st.spinner("Recording for 3 seconds..."):
                    t0 = time.time()
                    audio = st.session_state.detector.record_audio(duration=3.0)
                    t1 = time.time()
                    emotion, conf, lang = st.session_state.detector.process_audio(audio)
                    t2 = time.time()
                    emo_latency = (t2 - t1) * 1000
                    st.session_state.emotion_history.append((emotion, conf))
                    st.session_state.timestamps.append(time.time())
                    st.session_state.emotion_latency.append(emo_latency)
                    st.success(f"Detected emotion: {emotion} (confidence: {conf:.2f}) | Latency: {emo_latency:.1f} ms")
            except Exception as e:
                st.error(f"Error during recording: {str(e)}")
    
    with col2:
        st.subheader("Text Input")
        text = st.text_input("Enter text for sentiment analysis:", "I am happy today!")
        if st.button("Analyze Sentiment"):
            try:
                t0 = time.time()
                sentiment, conf, text_emotion = st.session_state.checker.analyze_sentiment(text)
                t1 = time.time()
                sent_latency = (t1 - t0) * 1000
                st.session_state.sentiment_history.append((sentiment, conf))
                st.session_state.text_emotion_history.append(text_emotion)
                st.session_state.sentiment_latency.append(sent_latency)
                st.success(f"Sentiment: {sentiment:.2f} (confidence: {conf:.2f}) | Latency: {sent_latency:.1f} ms | Text Emotion: {text_emotion}")
                # Tone switching demo
                if st.session_state.emotion_history:
                    emotion, emo_conf = st.session_state.emotion_history[-1]
                    t2 = time.time()
                    tone = st.session_state.switcher.determine_tone(emotion, emo_conf, sentiment, conf)
                    t3 = time.time()
                    tone_latency = (t3 - t2) * 1000
                    st.session_state.tone_latency.append(tone_latency)
                    st.info(f"Tone: {tone} | Tone Switch Latency: {tone_latency:.1f} ms")
            except Exception as e:
                st.error(f"Error during sentiment analysis: {str(e)}")
    
    # Display current metrics
    st.subheader("Current Metrics")
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    with metrics_col1:
        if st.session_state.emotion_history:
            emotion, conf = st.session_state.emotion_history[-1]
            st.metric("Current Emotion (Audio)", emotion)
            st.metric("Confidence", f"{conf:.2f}")
            if st.session_state.emotion_latency:
                st.metric("Emotion Latency (ms)", f"{st.session_state.emotion_latency[-1]:.1f}")
                st.metric("Avg Emotion Latency (ms)", f"{np.mean(st.session_state.emotion_latency):.1f}")
    
    with metrics_col2:
        if st.session_state.sentiment_history:
            sentiment, conf = st.session_state.sentiment_history[-1]
            st.metric("Current Sentiment", f"{sentiment:.2f}")
            st.metric("Confidence", f"{conf:.2f}")
            if st.session_state.sentiment_latency:
                st.metric("Sentiment Latency (ms)", f"{st.session_state.sentiment_latency[-1]:.1f}")
                st.metric("Avg Sentiment Latency (ms)", f"{np.mean(st.session_state.sentiment_latency):.1f}")
    
    with metrics_col3:
        if st.session_state.tone_latency:
            st.metric("Tone Switch Latency (ms)", f"{st.session_state.tone_latency[-1]:.1f}")
            st.metric("Avg Tone Switch Latency (ms)", f"{np.mean(st.session_state.tone_latency):.1f}")
    
    with metrics_col4:
        if st.session_state.text_emotion_history:
            st.metric("Text Emotion", st.session_state.text_emotion_history[-1])
    
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
        
        # Text emotion plot
        if st.session_state.text_emotion_history:
            fig_text_emotion = go.Figure()
            text_emotions = st.session_state.text_emotion_history
            fig_text_emotion.add_trace(go.Scatter(
                x=list(range(len(text_emotions))),
                y=text_emotions,
                mode='lines+markers',
                name='Text Emotion',
                line=dict(color='magenta')
            ))
            fig_text_emotion.update_layout(
                title='Text Emotion History',
                xaxis_title='Sample',
                yaxis_title='Emotion',
                showlegend=True
            )
            st.plotly_chart(fig_text_emotion, use_container_width=True, key=f"text_emotion_chart_{time.time()}")
        
        # Latency plots
        if st.session_state.emotion_latency:
            fig_emo_lat = go.Figure()
            fig_emo_lat.add_trace(go.Scatter(
                x=list(range(len(st.session_state.emotion_latency))),
                y=st.session_state.emotion_latency,
                mode='lines+markers',
                name='Emotion Latency',
                line=dict(color='red')
            ))
            fig_emo_lat.update_layout(
                title='Emotion Latency History',
                xaxis_title='Sample',
                yaxis_title='Latency (ms)',
                showlegend=True
            )
            st.plotly_chart(fig_emo_lat, use_container_width=True, key=f"emo_lat_chart_{time.time()}")
        if st.session_state.sentiment_latency:
            fig_sent_lat = go.Figure()
            fig_sent_lat.add_trace(go.Scatter(
                x=list(range(len(st.session_state.sentiment_latency))),
                y=st.session_state.sentiment_latency,
                mode='lines+markers',
                name='Sentiment Latency',
                line=dict(color='orange')
            ))
            fig_sent_lat.update_layout(
                title='Sentiment Latency History',
                xaxis_title='Sample',
                yaxis_title='Latency (ms)',
                showlegend=True
            )
            st.plotly_chart(fig_sent_lat, use_container_width=True, key=f"sent_lat_chart_{time.time()}")
        if st.session_state.tone_latency:
            fig_tone_lat = go.Figure()
            fig_tone_lat.add_trace(go.Scatter(
                x=list(range(len(st.session_state.tone_latency))),
                y=st.session_state.tone_latency,
                mode='lines+markers',
                name='Tone Switch Latency',
                line=dict(color='purple')
            ))
            fig_tone_lat.update_layout(
                title='Tone Switch Latency History',
                xaxis_title='Sample',
                yaxis_title='Latency (ms)',
                showlegend=True
            )
            st.plotly_chart(fig_tone_lat, use_container_width=True, key=f"tone_lat_chart_{time.time()}")

if __name__ == "__main__":
    main() 