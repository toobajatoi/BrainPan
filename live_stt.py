import speech_recognition as sr
import time

class LiveSTT:
    def __init__(self, phrase_time_limit=2):
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
        self.phrase_time_limit = phrase_time_limit

    def listen_and_transcribe(self):
        with self.mic as source:
            print("[LiveSTT] Adjusting for ambient noise...")
            self.recognizer.adjust_for_ambient_noise(source)
            print("[LiveSTT] Listening for speech ({}s chunks)...".format(self.phrase_time_limit))
            while True:
                try:
                    audio = self.recognizer.listen(source, phrase_time_limit=self.phrase_time_limit)
                    print("[LiveSTT] Processing audio chunk...")
                    try:
                        text = self.recognizer.recognize_google(audio)
                        print(f"[LiveSTT] Transcript: {text}")
                        yield text
                    except sr.UnknownValueError:
                        print("[LiveSTT] Could not understand audio.")
                        yield ""
                    except sr.RequestError as e:
                        print(f"[LiveSTT] API error: {e}")
                        yield ""
                except KeyboardInterrupt:
                    print("[LiveSTT] Stopped by user.")
                    break 