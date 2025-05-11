import os
import requests
import pyttsx3

class TTSEngine:
    def __init__(self):
        self.elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY')
        self.voice_map = {
            'happy': 'cheerful',
            'gentle': 'calm',
            'neutral': 'neutral',
            'angry': 'serious',
            'sad': 'empathetic',
        }
        if not self.elevenlabs_api_key:
            self.engine = pyttsx3.init()

    def synthesize(self, text, tone='neutral', play_audio=True):
        if self.elevenlabs_api_key:
            return self._synthesize_elevenlabs(text, tone, play_audio)
        else:
            return self._synthesize_local(text, tone, play_audio)

    def _synthesize_elevenlabs(self, text, tone, play_audio):
        voice = self.voice_map.get(tone, 'neutral')
        url = 'https://api.elevenlabs.io/v1/text-to-speech/voice-id'  # Replace with your voice ID
        headers = {
            'xi-api-key': self.elevenlabs_api_key,
            'Content-Type': 'application/json',
        }
        payload = {
            'text': f'<speak><voice emotion="{voice}">{text}</voice></speak>',
            'voice_settings': {'stability': 0.5, 'similarity_boost': 0.5},
            'model_id': 'eleven_multilingual_v2',
        }
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            audio_data = response.content
            if play_audio:
                with open('output_tts.wav', 'wb') as f:
                    f.write(audio_data)
                try:
                    import simpleaudio as sa
                    wave_obj = sa.WaveObject.from_wave_file('output_tts.wav')
                    play_obj = wave_obj.play()
                    play_obj.wait_done()
                except ImportError:
                    print('[TTS] Install simpleaudio for playback.')
            return audio_data
        else:
            print(f'[TTS] ElevenLabs API error: {response.status_code} {response.text}')
            return None

    def _synthesize_local(self, text, tone, play_audio):
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 1.0)
        # Optionally, set voice based on tone
        self.engine.say(text)
        if play_audio:
            self.engine.runAndWait()
        return None 