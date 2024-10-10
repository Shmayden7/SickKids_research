# Imports:
import wave
import numpy as np
import matplotlib.pyplot as plt

# Constants:
#########################

def plotWave(file_name: str, start: int, increment: int, dir='data/audio/'):

    full_path = dir + file_name
    obj = wave.open(full_path, 'rb')

    sample_freq = obj.getframerate()
    n_samples = obj.getnframes()
    n_channels = obj.getnchannels()
    signal_wave = obj.readframes(-1)
    obj.close()

    t_audio_ms = (n_samples / sample_freq) * 1000

    signal_array = np.frombuffer(signal_wave, dtype=np.int16)

    # If the audio is stereo, convert it to mono by averaging the two channels
    if n_channels == 2:
        signal_array = signal_array.reshape((-1, 2)).mean(axis=1)
        n_samples = len(signal_array)

    times = np.linspace(0, t_audio_ms, num=n_samples)

    plt.figure()
    plt.plot(times, signal_array)
    plt.title(file_name[:-4])
    plt.ylabel('Signal Wave')
    plt.xlabel('Time (ms)')
    plt.xlim(0, t_audio_ms)

    # Adding vertical lines
    plt.axvline(x=start, color='y')  # Starting point
    for i in range(50):
        plt.axvline(x=(start + ((i + 1) * increment)), color='r')  # Increment

    plt.show()
