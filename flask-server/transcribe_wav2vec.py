import os
import json
import re
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import torchaudio
from analysis.json_helpers import process_whisper_json

# Load the wav2vec model and tokenizer
model_name = "facebook/wav2vec2-base-960h"
model_name = "facebook/wav2vec2-large-960h"
tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

def transcribe_wav(wav_path):
    waveform, sample_rate = torchaudio.load(wav_path)
    inputs = tokenizer(waveform.squeeze().numpy(), return_tensors="pt", padding="longest")

    with torch.no_grad():
        logits = model(inputs.input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)[0]

    return {"text": transcription}

def run_wav2vec(condition, person):
    # CHANGE THE WAV DIRECTORY
    wav_directory = f"data/audio/pbk/normals/frequency/2000/{person}"
    json_directory = f"data/json/wav2vec_m/{condition}/{person}"

    # Ensure the JSON directory exists
    os.makedirs(json_directory, exist_ok=True)

    # Function to extract the numerical value before the first underscore
    def extract_number(filename):
        match = re.match(r"(\d+)_", filename)
        return int(match.group(1)) if match else float('inf')

    # Get and sort the list of WAV files in the directory based on the extracted number
    wav_files = sorted(
        [f for f in os.listdir(wav_directory) if f.endswith(".wav")],
        key=extract_number
    )

    # Iterate over sorted WAV files
    for filename in wav_files:
        wav_path = os.path.join(wav_directory, filename)
        
        # Transcribe the audio
        result = transcribe_wav(wav_path)
        
        # Create the JSON file path
        json_path = os.path.join(json_directory, f"{os.path.splitext(filename)[0]}.json")
        
        # Save the transcript as a JSON file
        with open(json_path, "w") as json_file:
            json.dump(result, json_file, indent=4)

        print(f"Transcribed {filename} and saved to {json_path}")

    process_whisper_json(json_directory)

names = ['aCauchi', 'Claire', 'Cole', 'Daniel', 'Hillary', 'Lulia', 'Melissa', 'Micheal', 'Polonenko', 'rCauchi', 'Robel', 'Samsoor']

list = 1

for name in names:
    condition = f'L{list}_c2000'
    person = f'L{list}_{name}'
    run_wav2vec(condition, person)
