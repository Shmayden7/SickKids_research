import os
import json
import re
import wave
import numpy as np
import subprocess
from deepspeech import Model
from analysis.json_helpers import process_whisper_json  # Reusing the whisper JSON processing function
import simpleaudio as sa

# Paths to the DeepSpeech model files
model_file_path = 'data/models/deepspeech-0.9.3-models.pbmm'
scorer_file_path = 'data/models/deepspeech-0.9.3-models.scorer'

# Load the DeepSpeech model
model = Model(model_file_path)
model.enableExternalScorer(scorer_file_path)

def run_deepspeech(condition, person):
    # CHANGE THE WAV DIRECTORY
    wav_directory = f"data/audio/pbk/normals/base/{person}"
    json_directory = f"data/json/deepspeech/{condition}/{person}"

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

        # Read the audio file
        with wave.open(wav_path, 'r') as wav_file:
            frames = wav_file.getnframes()
            buffer = wav_file.readframes(frames)
            rate = wav_file.getframerate()

        # Convert audio to the required format
        audio = np.frombuffer(buffer, dtype=np.int16)

        # Transcribe the audio
        result = model.stt(audio)

        # Create the JSON file path
        json_path = os.path.join(json_directory, f"{os.path.splitext(filename)[0]}.json")

        # Save the transcript as a JSON file
        with open(json_path, "w") as json_file:
            json.dump({'text': result}, json_file, indent=4)

        print(f"Transcribed {filename} and saved to {json_path}")

    process_whisper_json(json_directory)

names = ['aCauchi', 'Claire', 'Cole', 'Daniel', 'Hillary', 'Lulia', 'Melissa', 'Micheal', 'Polonenko', 'rCauchi', 'Robel', 'Samsoor']
list = 1

for name in names:
    condition = f'L{list}_normal'
    person = f'L{list}_{name}'
    run_deepspeech(condition, person)

# Load and play the WAV file
wave_obj = sa.WaveObject.from_wave_file('/Users/ayden/Downloads/ding.wav')
play_obj = wave_obj.play()
play_obj.wait_done()  # Wait until sound has finished playing