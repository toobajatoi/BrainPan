import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
from typing import Tuple, Dict, List
from collections import deque
import threading
import queue
import langid  # Improved language detection
from langdetect import detect
import re
import numpy as np

class SentimentChecker:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        # Initialize with a lightweight model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Performance optimization
        self.max_length = 128  # Limit sequence length for faster processing
        self.batch_size = 1  # Process one at a time for real-time
        self.latency_history = deque(maxlen=100)
        
        # Initialize processing queues
        self.input_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        self.is_processing = False
        self.processing_thread = None
        
        # Start processing thread
        self.start_processing()
        
    def start_processing(self):
        """Start the background processing thread"""
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._process_stream)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
    def stop_processing(self):
        """Stop the background processing thread"""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join()
            
    def _process_stream(self):
        """Background thread for processing sentiment analysis"""
        while self.is_processing:
            try:
                if not self.input_queue.empty():
                    text = self.input_queue.get()
                    result = self._analyze_sentiment_internal(text)
                    self.result_queue.put(result)
            except Exception as e:
                print(f"Error in processing thread: {str(e)}")
            time.sleep(0.001)  # Small sleep to prevent CPU overload
            
    def _analyze_sentiment_internal(self, text):
        """Internal method for sentiment analysis with minimal latency"""
        try:
            start_time = time.time()
            
            # Tokenize with optimized parameters
            inputs = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
                
            # Get sentiment and confidence
            sentiment_idx = torch.argmax(probs).item()
            confidence = probs[sentiment_idx].item()
            
            # Map to sentiment score (-1 to 1)
            sentiment_score = 1.0 if sentiment_idx == 1 else -1.0
            sentiment_score *= confidence  # Scale by confidence
            
            # Calculate latency
            latency = (time.time() - start_time) * 1000
            self.latency_history.append(latency)
            
            return {
                'sentiment': 'positive' if sentiment_score > 0 else 'negative',
                'score': sentiment_score,
                'confidence': confidence,
                'latency': latency
            }
            
        except Exception as e:
            print(f"Error in sentiment analysis: {str(e)}")
            return {
                'sentiment': 'neutral',
                'score': 0.0,
                'confidence': 0.0,
                'latency': 0.0
            }
            
    def analyze_sentiment(self, text):
        """Analyze sentiment with streaming optimization"""
        try:
            # Add to processing queue
            self.input_queue.put(text)
            
            # Get result from queue with timeout
            try:
                result = self.result_queue.get(timeout=0.1)  # 100ms timeout
                return result['sentiment'], result['score'], result['confidence']
            except queue.Empty:
                print("Warning: Sentiment analysis timeout")
                return 'neutral', 0.0, 0.0
                
        except Exception as e:
            print(f"Error in analyze_sentiment: {str(e)}")
            return 'neutral', 0.0, 0.0
            
    def get_average_latency(self):
        """Get average latency from recent history"""
        if not self.latency_history:
            return 0.0
        return sum(self.latency_history) / len(self.latency_history) 