import wave
import json
from vosk import Model, KaldiRecognizer

model_path = "models/hi"  # path to vosk Hindi model
audio_path = "input.wav"

wf = wave.open(audio_path, "rb")
rec = KaldiRecognizer(Model(model_path), wf.getframerate())

results = []
while True:
    data = wf.readframes(4000)
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        results.append(json.loads(rec.Result()))
results.append(json.loads(rec.FinalResult()))

# Get all words and their timestamps
words = []
for res in results:
    if 'result' in res:
        words.extend(res['result'])

for word in words:
    print(word["word"], word["start"], word["end"])
