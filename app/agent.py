from config import *
from google.genai import types
from elevenlabs.client import ElevenLabs
from elevenlabs import play
from google import genai

class WakeMateAgent:
    def __init__(self, gemini_api_key, elevenlabs_api_key):
        self.gemini = genai.Client(api_key=gemini_api_key)
        self.elevenlab = ElevenLabs(api_key=elevenlabs_api_key)
        
    def generate_warning(self, prompt):
        warning = DEFAULT_WARNING
        try:
            prompt += " Keep the prompt concise and to the point, avoiding any unnecessary details and under 50 words."
            response = self.gemini.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                )
            )
            warning = response.text.strip()
        except Exception as e:
            print(f"Error generating prompt: {e}")
        
        return warning
        

    def text_to_speech(self, text):
        audio = self.elevenlab.text_to_speech.convert(
            text=text,
            voice_id="pwMBn0SsmN1220Aorv15",
            model_id="eleven_flash_v2",
            output_format="mp3_44100_128"
        )

        play(audio)