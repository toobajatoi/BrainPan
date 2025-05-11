import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
from typing import Tuple, Dict, List
from collections import deque
import threading
import queue
from langdetect import detect

class SentimentChecker:
    def __init__(self, cache_size=100, smoothing_window=3):
        self.models = {}
        self.tokenizers = {}
        self.latency_history = deque(maxlen=100)
        self.smoothing_window = smoothing_window
        self.recent_predictions = deque(maxlen=smoothing_window)
        self.text_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        self.is_processing = False
        self.processing_thread = None
        self._load_models()
        if not torch.cuda.is_available():
            torch.set_num_threads(1)
        self.start_processing()

    def _load_models(self):
        try:
            en_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            self.tokenizers['en'] = AutoTokenizer.from_pretrained(
                en_model_name,
                model_max_length=128,
                truncation=True
            )
            self.models['en'] = AutoModelForSequenceClassification.from_pretrained(
                en_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            ).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')).eval()
            ur_model_name = "urduhack/bert-base-urdu-sentiment-analysis"
            self.tokenizers['ur'] = AutoTokenizer.from_pretrained(
                ur_model_name,
                model_max_length=128,
                truncation=True
            )
            self.models['ur'] = AutoModelForSequenceClassification.from_pretrained(
                ur_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            ).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')).eval()
        except Exception as e:
            print(f"Error loading models: {str(e)}")

    def start_processing(self):
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._process_text_stream)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def stop_processing(self):
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join()

    def _detect_language(self, text: str) -> str:
        try:
            lang = detect(text)
            if lang.startswith('ur'):
                return 'ur'
            return 'en'
        except Exception:
            return 'en'

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

    def _process_single_text(self, text: str) -> Tuple[float, float, str, str]:
        lang = self._detect_language(text)
        if lang not in self.models:
            lang = 'en'
        try:
            start_time = time.time()
            tokenizer = self.tokenizers[lang]
            model = self.models[lang]
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
                if lang == 'en':
                    sentiment = probs[0][2].item() - probs[0][0].item()
                    confidence = max(probs[0][0].item(), probs[0][2].item())
                    # Map to emotion label based on highest probability
                    pred_idx = torch.argmax(probs).item()
                    if pred_idx == 2:
                        emotion = 'happy'
                    elif pred_idx == 0:
                        emotion = 'angry'  # or 'sad', but we use 'angry' for strong negative
                    else:
                        emotion = 'neutral'
                else:
                    sentiment = probs[0][1].item() - probs[0][0].item()
                    confidence = max(probs[0][0].item(), probs[0][1].item())
                    pred_idx = torch.argmax(probs).item()
                    if pred_idx == 1:
                        emotion = 'happy'
                    elif pred_idx == 0:
                        emotion = 'sad'
                    else:
                        emotion = 'neutral'
                latency = (time.time() - start_time) * 1000
                self.latency_history.append(latency)
                return sentiment, confidence, emotion, lang
        except Exception as e:
            print(f"Error processing text: {str(e)}")
            return 0.0, 0.0, 'neutral', lang

    def analyze_sentiment(self, text: str) -> Tuple[float, float, str, str]:
        try:
            self.text_queue.put(text)
            try:
                sentiment, confidence, emotion, lang = self.result_queue.get(timeout=0.1)
                self.recent_predictions.append((sentiment, confidence, emotion, lang))
                if len(self.recent_predictions) == self.smoothing_window:
                    sentiments = [p[0] for p in self.recent_predictions]
                    confidences = [p[1] for p in self.recent_predictions]
                    emotions = [p[2] for p in self.recent_predictions]
                    langs = [p[3] for p in self.recent_predictions]
                    weights = [c for c in confidences]
                    total_weight = sum(weights)
                    if total_weight > 0:
                        sentiment = sum(s * w for s, w in zip(sentiments, weights)) / total_weight
                        confidence = sum(confidences) / len(confidences)
                        emotion = max(set(emotions), key=emotions.count)
                        lang = max(set(langs), key=langs.count)
                return sentiment, confidence, emotion, lang
            except queue.Empty:
                print("Warning: Processing timeout")
                return 0.0, 0.0, 'neutral', 'en'
        except Exception as e:
            print(f"Error in analyze_sentiment: {str(e)}")
            return 0.0, 0.0, 'neutral', 'en'

    def get_average_latency(self) -> float:
        if not self.latency_history:
            return 0.0
        return sum(self.latency_history) / len(self.latency_history) 