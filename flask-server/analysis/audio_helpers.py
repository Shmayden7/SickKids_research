#########################
# Imports:
import wave
import os
import pyaudio
import wave
# import webrtcvad as VAD
import audioop
from scipy.io import wavfile
import numpy as np
from scipy.signal import butter, filtfilt, freqz, lfilter
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from pydub import AudioSegment
#########################

##############################
# Description: Performs Voice Activity Detection (VAD) on a .wav audio file and prints speech detection results.
# Parameters:
#   - file_name: The name of the .wav audio file to process.
#   - dir: Optional; the directory containing the audio file (default is 'data/audio/').
# Returns: None
##############################
def waveVAD(file_name: str, dir='data/audio/') -> None:
    full_path = dir + file_name
    pa = pyaudio.PyAudio()

    # Open the .wav file
    wf = wave.open(full_path, "rb")
    print(wf.getframerate())

    # Set up the VAD algorithm
    vad = VAD.Vad()
    vad.set_mode(3)  # Aggressiveness (0-3)

    # Read the .wav file in chunks
    chunk_size = 320  # 320 => 20 ms
    while True:
        data = wf.readframes(chunk_size)
        if len(data) == 0:
            break

        # Convert the audio data to a format that VAD can process
        pcm_data, _ = audioop.ratecv(data, 2, 1, wf.getframerate(), 16000, None)

        # Detect speech using VAD
        is_speech = vad.is_speech(pcm_data, sample_rate=16000)

        # Do something with the speech detection result
        if is_speech:
            print("Speech detected")
        else:
            print("No speech detected")

##############################
# Description: Applies a low-pass filter to a stereo WAV audio file and saves the filtered output.
# Parameters: None
# Returns: None
##############################
def lowPassFilter():
    output_file = 'analysis/filtered_signal.wav'
    word = 'great'
    order = 12 # generally a max of 5 works best
    cutoff = 8000

    sample_rate, stereo_data = wavfile.read('analysis/1_L1_aCauchi.wav')
    duration = len(stereo_data)/sample_rate
    # Convert stereo audio to mono
    mono_data = stereo_data.mean(axis=1)

    # Applying the low pass filter to the data
    b, a = butter(order, cutoff, fs=sample_rate, btype='low', analog=False)
    filtered_data = lfilter(b, a, mono_data)

    # Save the filtered data as a WAV file
    wavfile.write(output_file, sample_rate, filtered_data)

    fft_data = np.fft.fft(mono_data)
    magnitude_spectrum = np.abs(fft_data)
    freq_axis = np.fft.fftfreq(len(mono_data), d=1/sample_rate)

    plt.plot(freq_axis, magnitude_spectrum)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.title('Magnitude Spectrum')
    plt.grid(True)
    plt.show()


    # # Plot the original data and the filtered data
    # plt.subplot(2, 2, 1)
    # t = np.linspace(0, duration, len(mono_data))
    # plt.plot(t, mono_data, 'b-', label='Original data')
    # plt.plot(t, filtered_data, 'g-', label='Filtered data')
    # plt.title(f"WAV Representation of: {word}")
    # plt.xlabel('Time [sec]')
    # plt.ylabel('Amplitude')
    # plt.legend()
    # plt.grid()

    # # Plot the frequency response of the lowpass filter
    # plt.subplot(2, 2, 2)
    # w, h = freqz(b, a, fs=sample_rate, worN=8000)
    # plt.plot(w, np.abs(h), 'b')
    # plt.plot(cutoff, 0.5 * np.sqrt(2), 'ko')
    # plt.axvline(cutoff, color='k')
    # plt.xlim(0, 0.5 * sample_rate)
    # plt.axvline(x=20000, color='r', linestyle='--')
    # plt.vlines(x=20000, ymin=0, ymax=1, colors='red', label='Normal Hearing Threshold')
    # plt.title(f"Lowpass Filter Frequency Response, Order: {order}")
    # plt.xlabel('Frequency [Hz]')
    # plt.grid()

    # # Plot the spectrogram pre filter
    # plt.subplot(2, 2, 3)
    # # plt.specgram(mono_data, Fs=sample_rate, NFFT=1024, noverlap=512)
    # _, _, _, im = plt.specgram(mono_data, Fs=sample_rate, NFFT=1024, noverlap=512)
    # plt.title("Spectrogram - Pre Filter")
    # plt.xlabel('Time [sec]')
    # plt.ylabel('Frequency [Hz]')
    # plt.colorbar(im).set_label('Intensity [dB]')

    # # Plot the spectrogram post filter
    # plt.subplot(2, 2, 4)
    # # plt.specgram(filtered_data, Fs=sample_rate, NFFT=1024, noverlap=512)
    # _, _, _, im = plt.specgram(filtered_data, Fs=sample_rate, NFFT=1024, noverlap=512)
    # plt.title("Spectrogram - Post Filter")
    # plt.xlabel('Time [sec]')
    # plt.ylabel('Frequency [Hz]')
    # plt.colorbar(im).set_label('Intensity [dB]')

    # plt.subplots_adjust(hspace=0.5)
    # plt.legend()
    # plt.show()

##############################
# Description: Applies a band-pass filter to a stereo WAV audio file and saves the filtered output, including plots of the original and filtered signals.
# Parameters: None
# Returns: None
##############################
def bandPassFilter():
    output_file = 'analysis/filtered_signal.wav'
    word = 'great'
    order = 12 # generally a max of 5 works best
    cutoff_low = 1000
    cutoff_high = 10000

    sample_rate, stereo_data = wavfile.read('analysis/1_L1_aCauchi.wav')
    duration = len(stereo_data)/sample_rate
    # Convert stereo audio to mono
    mono_data = stereo_data.mean(axis=1)

    # Applying the low pass filter to the data
    b, a = butter(order, [cutoff_low, cutoff_high], fs=sample_rate, btype='band', analog=False)
    filtered_data = lfilter(b, a, mono_data)

    # Save the filtered data as a WAV file
    wavfile.write(output_file, sample_rate, filtered_data)

    # Plot the original data and the filtered data
    plt.subplot(2, 2, 1)
    t = np.linspace(0, duration, len(mono_data))
    plt.plot(t, mono_data, 'b-', label='Original data')
    plt.plot(t, filtered_data, 'g-', label='Filtered data')
    plt.title(f"WAV Representation of: {word}")
    plt.xlabel('Time [sec]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()

    # Plot the frequency response of the lowpass filter
    plt.subplot(2, 2, 2)
    w, h = freqz(b, a, fs=sample_rate, worN=8000)
    plt.plot(w, np.abs(h), 'b')
    plt.plot([cutoff_low, cutoff_high], [0.5 * np.sqrt(2), 0.5 * np.sqrt(2)], 'ko')
    plt.axvline(cutoff_low, color='k')
    plt.axvline(cutoff_high, color='k')
    plt.xlim(0, 0.5 * sample_rate)
    plt.axvline(x=20000, color='r', linestyle='--')
    plt.vlines(x=20000, ymin=0, ymax=1, colors='red', label='Normal Hearing Threshold')
    plt.title(f"Lowpass Filter Frequency Response, Order: {order}")
    plt.xlabel('Frequency [Hz]')
    plt.legend()
    plt.grid()


    # Plot the spectrogram pre filter
    plt.subplot(2, 2, 3)
    # plt.specgram(mono_data, Fs=sample_rate, NFFT=1024, noverlap=512)
    _, _, _, im = plt.specgram(mono_data, Fs=sample_rate, NFFT=1024, noverlap=512)
    plt.title("Spectrogram - Pre Filter")
    plt.xlabel('Time [sec]')
    plt.ylabel('Frequency [Hz]')
    plt.colorbar(im).set_label('Intensity [dB]')

    # Plot the spectrogram post filter
    plt.subplot(2, 2, 4)
    # plt.specgram(filtered_data, Fs=sample_rate, NFFT=1024, noverlap=512)
    _, _, _, im = plt.specgram(filtered_data, Fs=sample_rate, NFFT=1024, noverlap=512)
    plt.title("Spectrogram - Post Filter")
    plt.xlabel('Time [sec]')
    plt.ylabel('Frequency [Hz]')
    plt.colorbar(im).set_label('Intensity [dB]')

    plt.subplots_adjust(hspace=0.5)
    plt.legend()
    plt.show()

##############################
# Description: Segments a WAV audio file into smaller parts based on specified start time and increments, saving each segment as a new WAV file.
# Parameters:
#     - file_name: Name of the input audio file (str).
#     - start_time_ms: Start time in milliseconds for segmentation (int).
#     - increment_time_ms: Increment time in milliseconds for each segment (int).
#     - n_segments: Number of segments to create (int).
#     - audio_path: Path to the input audio files (default is 'data/audio/').
#     - output_path: Path to save the segmented audio files (default is 'data/audio/segment/').
# Returns: None
##############################
def segment_pbk_recording(file_name:str, start_time_ms:int, increment_time_ms:int, n_segments: int, audio_path='data/audio/', output_path='data/audio/segment/') -> None:
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    print(audio_path)
    
    # Full path of the input audio file
    full_path = os.path.join(audio_path, file_name)
    
    # Open the input audio file
    with wave.open(full_path, 'rb') as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        
        # Read the entire audio file
        audio_data = wf.readframes(n_frames)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        frames_per_ms = sample_rate / 1000
        
        # Calculate the starting frame
        start_frame = int(start_time_ms * frames_per_ms)
        increment_frames = int(increment_time_ms * frames_per_ms)
        
        segment_number = 0
        current_frame = start_frame
        
        for i in range(n_segments):
            end_frame = current_frame + increment_frames
            
            # Handle the last segment case
            if end_frame > n_frames:
                end_frame = n_frames
            
            # Get the segment audio data
            segment_audio_array = audio_array[current_frame * n_channels:end_frame * n_channels]
            
            # Create the output file name
            segment_file_name = f"{segment_number}_{os.path.splitext(file_name)[0]}.wav"
            segment_file_path = os.path.join(output_path, segment_file_name)
            
            # Write the segment audio data to a new WAV file
            with wave.open(segment_file_path, 'wb') as segment_wf:
                segment_wf.setnchannels(n_channels)
                segment_wf.setsampwidth(sampwidth)
                segment_wf.setframerate(sample_rate)
                segment_wf.writeframes(segment_audio_array.tobytes())
            
            print(f"Segment {segment_number + 1} saved as {segment_file_name}")
            
            # Move to the next segment
            current_frame = end_frame
            segment_number += 1
            
            # Break if we've reached the end of the file
            if current_frame >= n_frames:
                break
                
    print(f"Segmented {file_name} into {n_segments} parts starting from {start_time_ms} ms with {increment_time_ms} ms increments and saved to {output_path}")