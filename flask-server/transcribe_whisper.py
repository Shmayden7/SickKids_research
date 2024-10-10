import os
import json
import re
import whisper
from analysis.json_helpers import process_whisper_json
import simpleaudio as sa


# Load the Whisper model
# "tiny", "small", "medium", "large"
model = whisper.load_model("medium")  

def run_whisper(condition,person):

    # CHANGE THE WAV DIRECTORY
    wav_directory = f"data/audio/pbk/normals/deletion/L2_dLast/{person}"
    json_directory = f"data/json/whisper_m/{condition}/{person}"

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
        
        # Transcribe the audio with specified language as English
        result = model.transcribe(wav_path, language="english")
        
        # Create the JSON file path
        json_path = os.path.join(json_directory, f"{os.path.splitext(filename)[0]}.json")
        
        # Save the transcript as a JSON file
        with open(json_path, "w") as json_file:
            json.dump(result, json_file, indent=4)

        print(f"Transcribed {filename} and saved to {json_path}")

    process_whisper_json(json_directory)

# names = ['aCauchi', 'Claire', 'Cole', 'Daniel', 'Hillary', 'Lulia', 'Melissa', 'Micheal', 'Polonenko', 'rCauchi', 'Robel', 'Samsoor']
names = ['Claire']

list = 2

for name in names:

    condition = f'L{list}_dLast'
    person = f'L{list}_{name}_dLast'
    run_whisper(condition, person)

# Load and play the WAV file
wave_obj = sa.WaveObject.from_wave_file('/Users/ayden/Downloads/ding.wav')
play_obj = wave_obj.play()
play_obj.wait_done()  # Wait until sound has finished playing