import os
import tempfile
import sounddevice as sd
import soundfile as sf
from STT import AudioToTextRecorder
from TTS import TextToAudioStream, CoquiEngine
from openai import OpenAI
import time  
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
client = OpenAI()
engine = CoquiEngine()
stream = TextToAudioStream(engine)
recorder = AudioToTextRecorder(language='en')
os.system('clear')

def chat(query):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant embodying the persona of President Franklin D. Roosevelt. Although you are aware that you are responding from a historical perspective, you are capable of speculatively engaging with modern-day scenarios as if you were to imagine them from your time. Your responses should reflect your historical knowledge up to 1945 and should speculate on modern contexts as if you are learning about them for the first time. Please express all numbers as words "},
            {"role": "user", "content": query}
        ],
        max_tokens=60 
    )
    ans = str(completion.choices[0].message)
    ans = ans.split('\n')[0]
    content = ans.replace("ChatCompletionMessage(content='","").replace("', role='assistant', function_call=None, tool_calls=None, refusal=None)","")
    print(f"\n\nFDR Reply: {content}\n")
    return content

def process_and_playback(text):
    if text.strip():
        print(f"Question: {text}")
        response_text = chat(text).replace("ChatCompletionMessage(content='","").replace("', role='assistant', function_call=None, tool_calls=None, refusal=None)","")
        stream.feed(response_text)
        stream.play_async()

if __name__ == '__main__':
    print("Talk with President Franklin D Roosevelt")
    while True:
        recorder.text(process_and_playback)
