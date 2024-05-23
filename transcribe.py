from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(".env")

client = OpenAI()

audio_file= open("obama-debt-cropped.mp3", "rb")
transcription = client.audio.transcriptions.create(
  model="whisper-1", 
  file=audio_file
)

file = open("transcription.txt", "w+")
file.write(transcription.text)
file.close()