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

class SentimentChecker:
    def __init__(self, cache_size=100, smoothing_window=3):
        # Initialize language detection patterns first
        self.urdu_patterns = [
            r'\b(kia|kya|kyaa|kiaa)\b',
            r'\b(ho|hoo|hoon)\b',
            r'\b(haan|ha|han)\b',
            r'\b(nahi|nahin|na)\b',
            r'\b(mein|main)\b',
            r'\b(tum|aap)\b',
            r'\b(kaise|kaisa|kaisi)\b',
            r'\b(kaisa|kaisi|kaise)\b',
            r'\b(kar|karo|karein)\b',
            r'\b(rehay|rahe|raha|rahi)\b',
            r'\b(tha|thi|the)\b',
            r'\b(hai|hain)\b',
            r'\b(mujhe|mujhko)\b',
            r'\b(tumhe|tumko)\b',
            r'\b(usko|unko)\b',
            r'\b(mera|meri|mere)\b',
            r'\b(tumhara|tumhari|tumhare)\b',
            r'\b(uska|uski|unke)\b'
        ]
        self.urdu_pattern = re.compile('|'.join(self.urdu_patterns), re.IGNORECASE)
        self.lang_map = {
            'en': 'English',
            'ur': 'Urdu',
            'roman_ur': 'Roman Urdu',
            'und': 'Unknown',
        }
        
        # Initialize model-related attributes
        self.models = {}
        self.tokenizers = {}
        self.latency_history = deque(maxlen=100)
        self.smoothing_window = smoothing_window
        self.recent_predictions = deque(maxlen=smoothing_window)
        self.text_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        self.is_processing = False
        self.processing_thread = None
        self.english_only = True  # Only English model is used for speed
        
        # Load models immediately
        print("Loading sentiment analysis model...")
        self._load_models()
        
        # Warm up the model
        print("Warming up sentiment model...")
        self._warm_up_model()
        
        if not torch.cuda.is_available():
            torch.set_num_threads(1)
        self.start_processing()

    def _load_models(self):
        try:
            # Use fast English sentiment model
            en_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
            self.tokenizers['en'] = AutoTokenizer.from_pretrained(
                en_model_name,
                model_max_length=128,
                truncation=True
            )
            self.models['en'] = AutoModelForSequenceClassification.from_pretrained(
                en_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            ).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')).eval()
        except Exception as e:
            print(f"Error loading models: {str(e)}")

    def _warm_up_model(self):
        """Warm up the model with some example texts"""
        warm_up_texts = [
            "I am very happy today!",
            "This is terrible news.",
            "I'm feeling okay, just normal."
        ]
        for text in warm_up_texts:
            try:
                self._process_single_text(text)
            except Exception as e:
                print(f"Warning: Error during model warm-up: {str(e)}")
        print("Sentiment model warm-up complete.")

    def start_processing(self):
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._process_text_stream)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def stop_processing(self):
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join()

    def _roman_urdu_rule_based(self, text: str) -> tuple:
        text = text.lower()
        happy_words = [
            'khush', 'acha', 'theek', 'mast', 'mazay', 'shukriya', 'mubarak', 'pyara', 'love', 'hansi', 'hans', 'masti', 'zabardast', 'best', 'great', 'amazing', 'awesome'
        ]
        sad_words = [
            'udaas', 'bura', 'buri', 'rona', 'ro', 'afsos', 'tanhai', 'akela', 'thak', 'thaka', 'thaki', 'thakay', 'thakawat', 'dard', 'takleef', 'problem', 'issue', 'parishan', 'parishani', 'naraaz', 'naraz', 'gussa', 'gham', 'bewafa', 'bewafai', 'hurt', 'cry', 'crying', 'sad', 'depressed', 'depression', 'toota', 'tooti', 'tootay', 'toot', 'fail', 'faili', 'faila', 'failay'
        ]
        angry_words = [
            'gussa', 'naraz', 'naraaz', 'badmash', 'badtameez', 'fight', 'jhagra', 'jhagda', 'larta', 'larti', 'lartay', 'shor', 'chilla', 'chillana', 'chillayi', 'chillata', 'chillati', 'chillatay', 'shor', 'shor machana', 'shor machaya', 'shor machayi', 'shor machate', 'shor machati', 'shor machatay', 'angry', 'anger', 'rage', 'furious', 'fight', 'fighting', 'shouting', 'shout', 'shouted', 'shouting', 'yell', 'yelling', 'yelled'
        ]
        score = 0
        for word in happy_words:
            if word in text:
                score += 1
        for word in sad_words:
            if word in text:
                score -= 1
        for word in angry_words:
            if word in text:
                score -= 2
        if score > 0:
            return {"sentiment": "positive", "score": 0.8}, 1.0, 'happy', 'roman_ur', 'Roman Urdu detected. Using rule-based sentiment/emotion.'
        elif score < -1:
            return {"sentiment": "negative", "score": -0.8}, 1.0, 'angry', 'roman_ur', 'Roman Urdu detected. Using rule-based sentiment/emotion.'
        elif score < 0:
            return {"sentiment": "negative", "score": -0.5}, 1.0, 'sad', 'roman_ur', 'Roman Urdu detected. Using rule-based sentiment/emotion.'
        else:
            return {"sentiment": "neutral", "score": 0.0}, 0.8, 'neutral', 'roman_ur', 'Roman Urdu detected. Using rule-based sentiment/emotion.'

    def _detect_language(self, text: str) -> str:
        try:
            if self.urdu_pattern.search(text):
                return 'roman_ur'
            lang, _ = langid.classify(text)
            if lang == 'ur':
                return 'ur'
            return lang
        except Exception:
            if self.urdu_pattern.search(text):
                return 'roman_ur'
            return 'und'

    def _process_text_stream(self):
        while self.is_processing:
            try:
                if not self.text_queue.empty():
                    text = self.text_queue.get()
                    result = self._process_single_text(text)
                    self.result_queue.put(result)
            except Exception as e:
                print(f"Error in processing thread: {str(e)}")
            time.sleep(0.001)

    def _process_single_text(self, text: str) -> Tuple[Dict, float, str, str]:
        lang = self._detect_language(text)
        if lang == 'roman_ur':
            sentiment, confidence, emotion, lang, warning = self._roman_urdu_rule_based(text)
            self.last_warning = warning
            return sentiment, confidence, emotion, lang
        model_key = 'en'
        try:
            start_time = time.time()
            tokenizer = self.tokenizers[model_key]
            model = self.models[model_key]
            
            # Pre-allocate tensors for common input sizes
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.inference_mode():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                pred_idx = torch.argmax(probs).item()
                score = probs[0][pred_idx].item()
                
                # Apply confidence threshold
                if score < 0.6:  # Lower threshold for sentiment
                    sentiment = {"sentiment": "neutral", "score": 0.0}
                    emotion = 'neutral'
                else:
                    if pred_idx == 1:
                        sentiment = {"sentiment": "positive", "score": float(score)}
                        emotion = 'happy'
                    elif pred_idx == 0:
                        sentiment = {"sentiment": "negative", "score": -float(score)}
                        emotion = 'sad'
                    else:
                        sentiment = {"sentiment": "neutral", "score": 0.0}
                        emotion = 'neutral'
                
                latency = (time.time() - start_time) * 1000
                self.latency_history.append(latency)
                print(f"[SentimentChecker] Latency: {latency:.2f} ms | Sentiment: {sentiment['sentiment']}, Score: {sentiment['score']:.2f}")
                self.last_warning = None
                return sentiment, score, emotion, lang
        except Exception as e:
            print(f"Error processing text: {str(e)}")
            self.last_warning = None
            return {"sentiment": "neutral", "score": 0.0}, 0.0, 'neutral', lang

    def analyze_sentiment(self, text: str) -> Tuple[Dict, float, str, str, str]:
        try:
            self.text_queue.put(text)
            try:
                sentiment, confidence, emotion, lang = self.result_queue.get(timeout=2.0)
                warning = getattr(self, 'last_warning', None)
                self.recent_predictions.append((sentiment, confidence, emotion, lang, warning))
                if len(self.recent_predictions) == self.smoothing_window:
                    sentiments = [p[0] for p in self.recent_predictions]
                    confidences = [p[1] for p in self.recent_predictions]
                    emotions = [p[2] for p in self.recent_predictions]
                    langs = [p[3] for p in self.recent_predictions]
                    warnings = [p[4] for p in self.recent_predictions if p[4]]
                    weights = [c for c in confidences]
                    total_weight = sum(weights)
                    if total_weight > 0:
                        # Weighted average for score
                        avg_score = sum(s["score"] * w for s, w in zip(sentiments, weights)) / total_weight
                        sentiment = {"sentiment": max(set([s["sentiment"] for s in sentiments]), key=[s["sentiment"] for s in sentiments].count), "score": avg_score}
                        confidence = sum(confidences) / len(confidences)
                        emotion = max(set(emotions), key=emotions.count)
                        lang = max(set(langs), key=langs.count)
                        warning = warnings[0] if warnings else None
                return sentiment, confidence, emotion, lang, warning
            except queue.Empty:
                print("Warning: Processing timeout")
                return {"sentiment": "neutral", "score": 0.0}, 0.0, 'neutral', 'und', None
        except Exception as e:
            print(f"Error in analyze_sentiment: {str(e)}")
            return {"sentiment": "neutral", "score": 0.0}, 0.0, 'neutral', 'und', None

    def get_average_latency(self) -> float:
        if not self.latency_history:
            return 0.0
        return sum(self.latency_history) / len(self.latency_history) 