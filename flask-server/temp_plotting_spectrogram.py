import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, get_window
from scipy.fft import fft
import soundfile as sf

def plot_spectrogram(audio, Fs, ax, title):
    Nspec = 1024
    noverlap = Nspec // 2
    f, t, Sxx = spectrogram(audio, fs=Fs, window=get_window('hann', Nspec), nperseg=Nspec, noverlap=noverlap)
    im = ax.pcolormesh(t, f / 1000, 10 * np.log10(Sxx), shading='gouraud')
    ax.set_ylabel('Frequency (kHz)')
    ax.set_xlabel('Time (s)')
    ax.set_title(title)
    return im

def plot_fft(audio, Fs, ax, title):
    N = len(audio)
    Y = fft(audio)
    f = np.linspace(0, Fs, N)
    ax.plot(f[:N//2], np.abs(Y[:N//2]) / N)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)

def main():
    # Change the file path to the audio file you want to debug
    audio_file = '/Users/ayden/Desktop/SickKids/audio/such.wav'
    
    audio, Fs = sf.read(audio_file)
    print(f"Audio - Sample Rate: {Fs}, Shape: {audio.shape}, Max Value: {np.max(audio)}, Min Value: {np.min(audio)}")
    
    if len(audio.shape) == 2:  # Stereo audio
        audio = audio[:, 0]  # Use only one channel
        print(f"Stereo Audio - Max Value: {np.max(audio)}, Min Value: {np.min(audio)}")
    
    fig, axs = plt.subplots(2, 1, figsize=(15, 10))

    cax = plot_spectrogram(audio, Fs, axs[0], 'Spectrogram')
    fig.colorbar(cax, ax=axs[0], orientation='vertical', label='Intensity (dB)')

    plot_fft(audio, Fs, axs[1], 'FFT')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()