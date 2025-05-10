import wave
import json
from vosk import Model, KaldiRecognizer

class STTEngine:
    def __init__(self, model_path="model/vosk-model-small-en-us-0.15"):  # Download Vosk model and put in 'model' folder
        self.model = Model(model_path)

    def transcribe_wav(self, wav_path: str, chunk_duration: float = 2.0):
        """
        Transcribe a WAV file in chunk_duration-second segments.
        Returns a list of (timestamp, transcript).
        """
        wf = wave.open(wav_path, "rb")
        sample_rate = wf.getframerate()
        chunk_size = int(sample_rate * chunk_duration)
        rec = KaldiRecognizer(self.model, sample_rate)
        results = []
        timestamp = 0.0
        while True:
            data = wf.readframes(chunk_size)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                transcript = res.get("text", "")
                results.append((timestamp, transcript))
            else:
                res = json.loads(rec.PartialResult())
                transcript = res.get("partial", "")
                results.append((timestamp, transcript))
            timestamp += chunk_duration
        wf.close()
        return results 