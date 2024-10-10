import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram, get_window

##############################
# Description: Converts stereo audio to mono by averaging channels; unchanged if mono.
# Parameters: audio (numpy.ndarray): Input audio array (1D for mono or 2D for stereo).
# Returns: numpy.ndarray: Converted mono audio array.
##############################
def stereo_to_mono(audio):
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    return audio

##############################
# Description: Normalizes audio by scaling to a range of -1 to 1.
# Parameters: audio (numpy.ndarray): Input audio array to normalize.
# Returns: numpy.ndarray: Normalized audio array.
##############################
def normalize_audio(audio):
    max_val = np.max(np.abs(audio))
    if max_val == 0:
        return audio
    return audio / max_val

##############################
# Description: Plots the spectrogram of audio data with specified parameters.
# Parameters: 
#   - audio (numpy.ndarray): Input audio signal.
#   - Fs (int): Sampling frequency of the audio.
#   - ax (matplotlib.axes.Axes): Axis to plot on.
#   - title_str (str): Title of the spectrogram.
#   - show_title (bool): Flag to show title; defaults to True.
# Returns: 
#   - cax: The color mesh object of the spectrogram.
##############################
def plot_spectrogram(audio, Fs, ax, title_str, show_title=True):
    Nspec = 1024
    noverlap = Nspec // 2
    f, t, Sxx = spectrogram(audio, fs=Fs, window=get_window('hann', Nspec), nperseg=Nspec, noverlap=noverlap)
    print(f"Spectrogram {title_str} - Frequency bins: {f.shape}, Time bins: {t.shape}, Sxx shape: {Sxx.shape}")
    cax = ax.pcolormesh(t, f / 1000, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='jet')
    if show_title:
        ax.set_title(title_str, fontsize=24, pad=20)
    ax.set_ylim([0, 10])
    ax.set_xlim([0, 0.7])
    ax.set_xticks(np.arange(0, 0.71, 0.2))
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xlabel('Time (s)' if ax == axs[1, 2] else '', fontsize=28, fontweight='bold', labelpad=20)
    if ax == axs[1, 0]:
        ax.set_ylabel('Frequency (kHz)', fontsize=28, fontweight='bold', labelpad=20)
    cax.set_clim(-40, 60)  # Set color scale limits
    return cax

##############################
# Description: Plots the audio waveform with time on the x-axis.
# Parameters: 
#   - audio (numpy.ndarray): Input audio signal.
#   - Fs (int): Sampling frequency of the audio.
#   - ax (matplotlib.axes.Axes): Axis to plot on.
#   - title_str (str): Title of the waveform.
# Returns: 
#   - None
##############################
def plot_waveform(audio, Fs, ax, title_str):
    t_audio = np.arange(len(audio)) / Fs
    ax.plot(t_audio, audio)
    ax.set_title(title_str, fontsize=24, fontweight='bold', pad=20)
    ax.set_ylim([-0.75, 1])
    ax.set_xlim([0, 0.7])
    ax.set_xticks(np.arange(0, 0.71, 0.2))
    ax.tick_params(axis='both', which='major', labelsize=16)
    if ax == axs[0, 0]:
        ax.set_ylabel('Amplitude\n(Arbitrary Units)', fontsize=28, fontweight='bold', labelpad=20)
    ax.set_xlabel('Time (s)' if ax == axs[0, 2] else '', fontsize=28, fontweight='bold', labelpad=20)

##############################
# Description: Plots the waveform of unaltered audio and its consonant deletions.
# Parameters: None
# Returns: None
##############################
def plot_deletions():
    fig, axs = plt.subplots(1, 3, figsize=(15, 7))

    # Load and process the unaltered audio file
    file_path = '/Users/ayden/Desktop/Sickkids/audio/such.wav'
    Fs, base = wavfile.read(file_path)
    base_waveform = stereo_to_mono(base)
    base_waveform = normalize_audio(base_waveform)

    # Plot unaltered waveform
    plot_waveform(base_waveform, Fs, axs[0], 'Unaltered')

    # First Consonant Deletion
    start_time = 0.0       # Starting time in seconds
    end_time = 0.26      # Ending time in seconds
    start_sample = int(start_time * Fs)
    end_sample = int(end_time * Fs)
    fDelBase = np.copy(base_waveform)
    fDelBase[start_sample:end_sample] = 0
    plot_waveform(fDelBase, Fs, axs[1], 'First Consonant Deletion')

    # Last Consonant Deletion
    start_time = 0.44       # Starting time in seconds
    end_time = 0.69      # Ending time in seconds
    start_sample = int(start_time * Fs)
    end_sample = int(end_time * Fs)
    lDelBase = np.copy(base_waveform)
    lDelBase[start_sample:end_sample] = 0
    plot_waveform(lDelBase, Fs, axs[2], 'Last Consonant Deletion')

    # Styling
    axs[1].set_xlabel('Time (s)', fontsize=30, fontweight='bold', labelpad=20)
    axs[0].set_ylabel('Amplitude\n(Arbitrary Units)', fontsize=30, fontweight='bold', labelpad=20)
    plt.tight_layout()
    plt.show()

##############################
# Description: Loads and processes audio files; plots waveforms and spectrograms.
# Parameters: None
# Returns: None
##############################
def main():
    global axs  # Ensure axs is available globally within the main function
    fig, axs = plt.subplots(2, 5, figsize=(20, 10))

    # Load and process the unaltered audio file
    file_path = '/Users/ayden/Desktop/Sickkids/audio/such.wav'
    Fs, base = wavfile.read(file_path)
    print(f"Unaltered Audio - Sample Rate: {Fs}, Shape: {base.shape}, Max Value: {np.max(base)}, Min Value: {np.min(base)}")

    # Process for waveform
    base_waveform = stereo_to_mono(base)
    base_waveform = normalize_audio(base_waveform)
    print(f"After processing for waveform - Shape: {base_waveform.shape}, Max Value: {np.max(base_waveform)}, Min Value: {np.min(base_waveform)}")
    plot_waveform(base_waveform, Fs, axs[0, 0], 'Unaltered')
    cax = plot_spectrogram(base[:, 0] if base.ndim == 2 else base, Fs, axs[1, 0], 'Unaltered', show_title=False)

    # Load and process the low-pass filtered audio files
    file_paths = [
        '/Users/ayden/Desktop/Sickkids/audio/such_6000.wav',
        '/Users/ayden/Desktop/Sickkids/audio/such_4000.wav',
        '/Users/ayden/Desktop/Sickkids/audio/such_2000.wav',
        '/Users/ayden/Desktop/Sickkids/audio/such_1000.wav'
    ]

    titles = [
        'LPF at 6 kHz',
        'LPF at 4 kHz',
        'LPF at 2 kHz',
        'LPF at 1 kHz'
    ]

    for i, (file_path, title) in enumerate(zip(file_paths, titles)):
        Fs, audio = wavfile.read(file_path)
        print(f"{title} - Sample Rate: {Fs}, Shape: {audio.shape}, Max Value: {np.max(audio)}, Min Value: {np.min(audio)}")

        # Process for waveform
        audio_waveform = stereo_to_mono(audio)
        audio_waveform = normalize_audio(audio_waveform)
        print(f"After processing for waveform {title} - Shape: {audio_waveform.shape}, Max Value: {np.max(audio_waveform)}, Min Value: {np.min(audio_waveform)}")

        # Plot waveform
        plot_waveform(audio_waveform, Fs, axs[0, i + 1], title)

        # Plot spectrogram
        cax = plot_spectrogram(audio[:, 0] if audio.ndim == 2 else audio, Fs, axs[1, i + 1], title, show_title=False)

    # Add colorbar to the right of the last column
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.86, 0.08, 0.02, 0.37])  # Adjust the position and size of the colorbar
    cbar = fig.colorbar(cax, cax=cbar_ax)
    cbar.set_label('Level (dB)', fontsize=24, fontweight='bold')
    cbar.ax.tick_params(labelsize=14)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

if __name__ == "__main__":
    main()
    plot_deletions()