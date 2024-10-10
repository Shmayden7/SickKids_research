import librosa
import numpy as np
import webrtcvad
import soundfile as sf
import os
import matplotlib.pyplot as plt

##############################
# Description: Combines audio files by extracting voiced segments using a Voice Activity 
#              Detector (VAD) and saves the result as a single audio file.
# Parameters: 
#   - files (list): List of paths to the audio files to be combined.
#   - vad_aggressiveness (int): Aggressiveness level for VAD (0-3).
#   - output_filename (str): The path where the combined audio file will be saved.
# Returns: None. Outputs the combined audio file at the specified location.
# Dependencies: webrtcvad, librosa, numpy, soundfile
##############################
def combine_audio_files(files, vad_aggressiveness, output_filename):
    vad = webrtcvad.Vad()
    vad.set_mode(vad_aggressiveness)  # Set aggressiveness level (0-3)

    combined_audio = []

    for filepath in files:
        # Load the audio file
        audio, sr = librosa.load(filepath, sr=16000)  # Ensure sample rate is 16000 Hz

        # Convert audio to 16-bit PCM format
        audio_pcm = (audio * 32768).astype(np.int16)

        # Function to extract frames
        def frame_generator(frame_duration_ms, audio, sample_rate):
            n = int(sample_rate * frame_duration_ms / 1000)
            offset = 0
            while offset + n <= len(audio):
                yield audio[offset:offset + n]
                offset += n

        # Function to detect speech frames
        def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
            num_padding_frames = int(padding_duration_ms / frame_duration_ms)
            ring_buffer = []
            triggered = False

            voiced_frames = []

            for frame in frames:
                is_speech = vad.is_speech(frame.tobytes(), sample_rate)

                if is_speech:
                    if not triggered:
                        triggered = True
                        voiced_frames.extend(ring_buffer)
                        ring_buffer.clear()
                    voiced_frames.append(frame)
                else:
                    if triggered:
                        ring_buffer.append(frame)
                        if len(ring_buffer) > num_padding_frames:
                            triggered = False
                            yield b''.join(voiced_frames)
                            ring_buffer.clear()
                            voiced_frames = []
                    else:
                        ring_buffer.append(frame)
                        if len(ring_buffer) > num_padding_frames:
                            ring_buffer.pop(0)

            if voiced_frames:
                yield b''.join(voiced_frames)

        # Generate frames and collect voiced segments
        frames = frame_generator(30, audio_pcm, sr)
        segments = vad_collector(sr, 30, 300, vad, frames)

        # Combine voiced segments
        voiced_audio = b''.join(segments)

        # Convert back to float
        voiced_audio_float = np.frombuffer(voiced_audio, dtype=np.int16).astype(np.float32) / 32768.0

        # Append to combined audio
        combined_audio.append(voiced_audio_float)

    # Concatenate all voiced segments into one audio file
    combined_audio = np.concatenate(combined_audio)

    # Save the combined audio file
    sf.write(output_filename, combined_audio, sr)
    print(f"Voiced segments combined and saved to {output_filename}")

##############################
# Description: Plots the audio waveform and its estimated fundamental frequency (F0) 
#              over time from a given WAV file. It also computes and displays the average 
#              fundamental frequency while filtering out frequencies below a specified lower bound.
# Parameters: 
#   - wav_file (str): The path to the input WAV audio file.
#   - frequency_lower_bound (float): The lower frequency threshold for filtering F0 values.
# Returns: 
#   - average_f0 (float): The average fundamental frequency (F0) calculated from the audio.
# Dependencies: librosa, numpy, matplotlib
##############################
def plot_f0(wav_file, frequency_lower_bound):
    file_name = os.path.basename(wav_file)
    audio, sr = librosa.load(wav_file, sr=16000)

    # Estimate the fundamental frequency
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio, 
        fmin=librosa.note_to_hz('C2'), 
        fmax=librosa.note_to_hz('C7'), 
        sr=sr
    )

    # Filter out frequencies below the lower bound
    f0_filtered = np.where(f0 < frequency_lower_bound, np.nan, f0)

    # Calculate the average fundamental frequency, ignoring NaN values
    average_f0 = np.nanmean(f0_filtered)

    # Generate time axis for audio and F0
    times_audio = librosa.times_like(audio, sr=sr)
    times_f0 = librosa.times_like(f0, sr=sr)

    plt.figure(figsize=(14, 6))

    # Plotting audio waveform
    plt.subplot(2, 1, 1)
    plt.plot(times_audio, audio, label='Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Audio Waveform: {file_name}')
    plt.xticks([])  # Turn off x-axis ticks

    # Plotting Fundamental
    plt.subplot(2, 1, 2)
    plt.plot(times_f0, f0_filtered, label='Fundamental Frequency (F0)', color='r')
    plt.axhline(y=average_f0, color='b', linestyle='--', label=f'Average F0: {average_f0:.2f} Hz')
    # plt.axhline(y=frequency_lower_bound, color='g', linestyle='--', label=f'Lower Bound: {frequency_lower_bound} Hz')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title(f'Fundamental Frequency Over Time: {file_name}')
    plt.xticks(np.arange(0, times_f0[-1] + 1, step=1))  # Set tick at every second
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"{file_name} F0: {average_f0:.2f} Hz")
    return average_f0

# # Example usage
# file_list = [
#     'data/audio/F0/L1_Samsoor.wav',
#     'data/audio/F0/L2_Samsoor.wav'
# ]
# vad_aggressiveness = 3  # Level of VAD aggressiveness
# output_file = 'data/audio/F0/combined/Cole.wav'
# # combine_audio_files(file_list, vad_aggressiveness, output_file)

female = [
    'Claire.wav',
    'Hillary.wav',
    'Lulia.wav',
    'Melissa.wav',
    'Polonenko.wav',
    'rCauchi.wav',
]

male = [
    'aCauchi.wav',
    'Cole.wav',
    'Daniel.wav',
    'Robel.wav',
    'Samsoor.wav',
    'Micheal.wav',
]
frequency_lower_bound = 75  # Hz

female_f0 = 0
male_f0 = 0
for male, female in zip(male, female):
    average_male_f0 = plot_f0(('data/audio/F0/combined/' + male), frequency_lower_bound)
    average_female_f0 = plot_f0(('data/audio/F0/combined/' + female), frequency_lower_bound)

    male_f0 += average_male_f0
    female_f0 += average_female_f0

print()
print(f'total male_f0: {round(male_f0/6,2)} Hz') 
print(f'total female_f0: {round(female_f0/6,2)} Hz') 
print(f'total f0: {round((female_f0/6 + male_f0/6)/2,2)} Hz') 
