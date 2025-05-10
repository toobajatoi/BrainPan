import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
from typing import Tuple, Dict, List
from collections import deque
import threading
import queue
import re

# Keyword lists for emotion detection
EMOTION_KEYWORDS = {
    'happy': [r"happy", r"joy", r"delighted", r"excited", r"glad", r"pleased", r"cheerful", r"smile", r"laugh", r"awesome", r"amazing", r"fantastic", r"great", r"wonderful"],
    'sad': [r"sad", r"unhappy", r"depressed", r"down", r"cry", r"crying", r"miserable", r"gloomy", r"tear", r"sorrow", r"heartbroken", r"blue", r"disappointed"],
    'angry': [r"angry", r"mad", r"furious", r"rage", r"annoyed", r"irritated", r"hate", r"pissed", r"frustrated", r"upset", r"what the hell", r"wrong with you", r"idiot", r"stupid", r"dumb", r"shut up", r"useless", r"disgusting", r"worst", r"terrible", r"awful", r"sucks", r"pathetic"],
    'depressed': [r"depressed", r"hopeless", r"worthless", r"empty", r"numb", r"can't go on", r"give up", r"no point", r"suicidal", r"lost all hope"]
}

# Map emotions to sentiment values
EMOTION_SENTIMENT = {
    'happy': (1.0, 1.0),
    'sad': (-0.7, 1.0),
    'angry': (-1.0, 1.0),
    'depressed': (-1.0, 1.0)
}

class SentimentChecker:
    def __init__(self, cache_size=100, smoothing_window=3):
        self.model = None
        self.tokenizer = None
        self.latency_history = deque(maxlen=100)
        self.smoothing_window = smoothing_window
        self.recent_predictions = deque(maxlen=smoothing_window)
        
        # Streaming processing setup
        self.text_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        self.is_processing = False
        self.processing_thread = None
        
        # Load model
        self.model = self._load_model()
        
        # Set number of threads for CPU inference
        if not torch.cuda.is_available():
            torch.set_num_threads(1)
            
        # Start processing thread
        self.start_processing()
        
    def _load_model(self):
        """Load a better model for sentiment analysis"""
        try:
            model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                model_max_length=128,
                truncation=True
            )
            
            # Load model with reduced precision
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            # Prepare model for quantization
            model.eval()
            model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            
            return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None

    def start_processing(self):
        """Start the background processing thread"""
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._process_text_stream)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def stop_processing(self):
        """Stop the background processing thread"""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join()

    def _process_text_stream(self):
        """Background thread for processing text chunks"""
        while self.is_processing:
            try:
                if not self.text_queue.empty():
                    text = self.text_queue.get()
                    result = self._process_single_text(text)
                    self.result_queue.put(result)
            except Exception as e:
                print(f"Error in processing thread: {str(e)}")
            time.sleep(0.001)  # Small sleep to prevent CPU overload

    def _process_single_text(self, text: str) -> Tuple[float, float, str]:
        """Process a single text chunk with minimal latency"""
        if self.model is None:
            return 0.0, 0.0, 'neutral'
            
        # Rule-based override for emotion keywords
        for emotion, patterns in EMOTION_KEYWORDS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    sentiment, confidence = EMOTION_SENTIMENT[emotion]
                    return sentiment, confidence, emotion
            
        try:
            start_time = time.time()
            
            # Fast tokenization
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Fast inference
            with torch.inference_mode():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                
                # For this model: 0=negative, 1=neutral, 2=positive
                sentiment = probs[0][2].item() - probs[0][0].item()  # positive - negative
                confidence = max(probs[0][0].item(), probs[0][2].item())
                
                # Assign emotion label based on sentiment
                if sentiment > 0.5:
                    emotion = 'happy'
                elif sentiment < -0.7:
                    emotion = 'angry'
                elif sentiment < -0.3:
                    emotion = 'sad'
                else:
                    emotion = 'neutral'
                
                # Calculate latency
                latency = (time.time() - start_time) * 1000
                self.latency_history.append(latency)
                
                return sentiment, confidence, emotion
                
        except Exception as e:
            print(f"Error processing text: {str(e)}")
            return 0.0, 0.0, 'neutral'

    def analyze_sentiment(self, text: str) -> Tuple[float, float, str]:
        """Analyze sentiment with streaming optimization"""
        if self.model is None:
            return 0.0, 0.0, 'neutral'
            
        try:
            # Add to processing queue
            self.text_queue.put(text)
            
            # Get result from queue with timeout
            try:
                sentiment, confidence, emotion = self.result_queue.get(timeout=0.1)  # 100ms timeout
                
                # Apply smoothing
                self.recent_predictions.append((sentiment, confidence, emotion))
                if len(self.recent_predictions) == self.smoothing_window:
                    sentiments = [p[0] for p in self.recent_predictions]
                    confidences = [p[1] for p in self.recent_predictions]
                    emotions = [p[2] for p in self.recent_predictions]
                    
                    # Weighted average based on confidence
                    weights = [c for c in confidences]
                    total_weight = sum(weights)
                    if total_weight > 0:
                        sentiment = sum(s * w for s, w in zip(sentiments, weights)) / total_weight
                        confidence = sum(confidences) / len(confidences)
                        # Most common emotion
                        emotion = max(set(emotions), key=emotions.count)
                
                return sentiment, confidence, emotion
                
            except queue.Empty:
                print("Warning: Processing timeout")
                return 0.0, 0.0, 'neutral'
                
        except Exception as e:
            print(f"Error in analyze_sentiment: {str(e)}")
            return 0.0, 0.0, 'neutral'

    def get_average_latency(self) -> float:
        """Get average latency from recent history"""
        if not self.latency_history:
            return 0.0
        return sum(self.latency_history) / len(self.latency_history) 