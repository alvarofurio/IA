import os
import ollama
import whisper

# Input de un fichero de audio que se encuentre en el directorio actual
audio = input("Escribe el nombre del archivo: ")
prompt = input("Escribe el prompt: ")
while not os.path.isfile(audio):
    print("\nEL fichero no existe en el directorio\n")
    audio = input("Escribe el nombre del archivo: ")

# Transcripción del audio por Whisper
model = whisper.load_model("base")
transcription = model.transcribe(audio)['text']

# Generación de respuesta con llama3
response = ollama.chat(model='llama3', messages=[
  {
    'role': 'user',
    'content':  transcription + "\n" + prompt,
  },
])
print(response['message']['content'])