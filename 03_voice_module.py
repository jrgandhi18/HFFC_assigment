import requests
import os
from dotenv import load_dotenv

load_dotenv()
SARVAM_KEY = os.getenv("SARVAM_API_KEY")

def transcribe_audio(audio_file_path):
    """
    Converts speech to text using Sarvam AI.
    """
    url = "https://api.sarvam.ai/speech-to-text"
    
    files = {'file': open(audio_file_path, 'rb')}
    headers = {"api-subscription-key": SARVAM_KEY}
    
    # Sarvam usually needs a prompt or model type
    data = {
        "model": "saarika:v1", 
        "language_code": "hi-IN" # Ye Hindi/English mix handle kar lega
    }

    response = requests.post(url, headers=headers, files=files, data=data)
    
    if response.status_status == 200:
        return response.json().get('transcript')
    else:
        return f"Error: {response.text}"

def speak_text(text, target_language="hi-IN"):
    """
    Converts text back to speech using Sarvam TTS.
    """
    url = "https://api.sarvam.ai/text-to-speech"
    
    payload = {
        "inputs": [text],
        "target_language_code": target_language,
        "speaker": "meera", # Voice selection
        "model": "bulbul:v1"
    }
    headers = {
        "Content-Type": "application/json",
        "api-subscription-key": SARVAM_KEY
    }

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        with open("response.wav", "wb") as f:
            f.write(response.content)
        print("AI is speaking now...")
    else:
        print("TTS Error:", response.text)