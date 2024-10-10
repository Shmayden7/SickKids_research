#########################
# Imports:
import os, glob
from pydub import AudioSegment
from pydub.playback import play
from scipy.signal import correlate
from sklearn.decomposition import FastICA
import matplotlib.pyplot as py_plt
import numpy as np

# Local Imports
from api.assembly_ai.assembly_helpers import runAssembly
from api.speechmatics.speechmatics_index import runSpeechmatics
from analysis.json_helpers import loadDictFromJSON

# Class Imports
from classes.DoubleyLinkedList import DoubleyLinkedList
#########################

class WavAudio():
    def __init__(self, file_name, cfg, dir='data/audio/'):
        # Init Properties
        self.dir = dir
        self.file_name = file_name
        self.file_path = dir + file_name
        self.cache = dir + file_name[:-4] + '/'
        self.master = None
        self.audio = AudioSegment.from_wav(self.file_path)
        self.cfg = cfg
        self.assembly_LL = DoubleyLinkedList()
        self.speechmatics_LL = DoubleyLinkedList()

        # Init Functions
        self.__make_cache()
        self.__classify_master_assembly()
        self.__init_Assembly_LL()

    #########################
    # Cache Functions
    def __make_cache(self):
        try:
            os.makedirs(self.cache, exist_ok=False)
            print('Made Cache for: {}'.format(self.file_name[:-4]))
        except OSError as e:
            print('The Cache for {} Already Exists.'.format(self.file_name[:-4]))

    def __files_in_cache(self):
        files = []
        for root, dirs, filenames in os.walk(self.cache):
            files.extend(filenames)
            break

        # Sort files based on numeric values in filenames
        #files = sorted(files, key=lambda x: int("".join([i for i in x if i.isdigit()])) if any(i.isdigit() for i in x) else x)
        files = sorted(files, key=lambda x: int("".join([i for i in x if i.isdigit()])) if any(i.isdigit() for i in x) else float('inf'))

        return files

    def clear_cache(self):
        # UNFINISHED!!! NEED TO DO THIS
        if os.path.isdir(self.cache):
            print('there is a cache')

            files = glob.glob(self.cache)
            for f in files:
                print(f)

    def __classify_master_assembly(self):
        contains_master = False
        files = self.__files_in_cache()
        for file_name in files: 
            if file_name == '{}.json'.format(self.file_name[:-4]):
                contains_master = True
        if not contains_master:
            print('Classifying Master Audio...')
            file_path = self.file_path[:-len(self.file_name)]
            runAssembly(self.file_name, self.cfg, file_path, self.cache)
        else:
            print('Master transcription present')
        self.master = '{}.json'.format(self.file_name[:-4])

    def classify_cache_assembly(self):
        files = self.__files_in_cache()
        for file_name in files:
            if file_name[-4:] == '.wav':
                runAssembly(file_name, self.cfg, self.cache, 'data/json/segmented_norms/')
        print('{} Cache Classified and dumped!'.format(self.file_name[:-4]))

    def classify_cache_speechmatics(self,dump_location):
        files = self.__files_in_cache()
        has_wav = False
        for file_name in files:
            if file_name[-4:] == '.wav':
                has_wav = True
                runSpeechmatics(file_name, self.cache, dump_location)
        if has_wav:
            print('{} Cache Classified and dumped!'.format(self.file_name[:-4]))
        else:
            print('Error, The Cashe is empty :(')

    #########################
    # Linked List Functions

    def __init_Assembly_LL(self):
        json = loadDictFromJSON(self.master,self.cache)

        for word in json["words"]:
            self.assembly_LL.push_tail(word)

    #########################
    # Helper Functions
    def get_duration(self):
        return self.audio.duration_seconds
    
    def play(self):
        play(self.audio)
    
    def segment(self, start_ms, end_ms, segment_file_name):
        split_audio = self.audio[start_ms:end_ms]
        split_audio.export(self.cache + segment_file_name, format="wav")
        print('Dumped {} to Cache'.format(segment_file_name))
        
    def n_segments_fixed(self, start: int, step: int, n_segments: int):
        end = ((step*n_segments)+start)
        index = 0
        for i in range(start, end, step):
            split_fn = str(index) + '_' + self.file_name
            self.segment(i, i+step, split_fn)
            index += 1

    def remove_audio_by_time_seg(self, segments: list):
        # this is needed because when you remove a chunck of audio from an audio segment
        # the time of the entire segment becomes shorter, doing it this way takes that into account
        removal_duration = 0
        for segment in segments:
            curr_segment_duration = segment[1] - segment[0]
            self.audio = self.audio[:(segment[0])-removal_duration] + self.audio[(segment[1])-removal_duration:]
            removal_duration += curr_segment_duration
        
    def remove_phrase_via_assembly_LL(self, phrase):
        words = phrase.split()
        curr = self.assembly_LL.head
        index_of_removal_words = []
        time_frames_to_remove = []
        
        start = 0
        end = 0
        for index in range(self.assembly_LL.get_size()):
            curr_word = curr.data['text']

            if curr_word == words[0]: # if weve hit the first word of the phrase
                #print("Beginning: {}".format(curr_word))
                start = curr.data['start']
                index_of_removal_words.append(index)
                curr = curr.next

            elif curr_word == words[len(words)-1]: # if weve hit the last word of the phrase
                #print("End: {}".format(curr_word))
                end = curr.data['end']
                time_frames_to_remove.append([start, end])
                index_of_removal_words.append(index)
                curr = curr.next
                start = 0
                end = 0

            elif curr_word in words:
                #print("Within: {}".format(curr_word))
                index_of_removal_words.append(index)
                curr = curr.next

            else: # if the word is not the first or last word of the phrase
                curr = curr.next

        # remove the the audio sections from the parent audio
        self.remove_audio_by_time_seg(time_frames_to_remove)

        # removing words in phease form the base LL
        number_of_nodes_removed = 0
        for index in index_of_removal_words:
            self.assembly_LL.remove_at(index-number_of_nodes_removed)
            number_of_nodes_removed += 1

        self.assembly_LL.print()

    def plot_audio_with_repeating_phrase_CC(self, repeating_phrase_audio_file):
        # Load the original long audio file
        original_audio = self.audio

        # Load the shorter audio file containing the repeating phrase
        repeating_phrase_audio = AudioSegment.from_file(repeating_phrase_audio_file)

        # Extract the raw audio data from the original long audio file
        original_audio_data = np.array(original_audio.get_array_of_samples())

        # Extract the raw audio data from the repeating phrase audio file
        repeating_phrase_audio_data = np.array(repeating_phrase_audio.get_array_of_samples())

        # Find the repeating phrase in the original audio data using cross-correlation
        cross_correlation = np.correlate(original_audio_data, repeating_phrase_audio_data, mode='same')

        # Find the start and end times of the repeating phrase
        repeating_phrase_start = np.argmax(cross_correlation)
        repeating_phrase_end = repeating_phrase_start + len(repeating_phrase_audio_data)

        # Convert time to seconds
        repeating_phrase_start_time = repeating_phrase_start / original_audio.frame_rate
        repeating_phrase_end_time = repeating_phrase_end / original_audio.frame_rate

        # Create a time array for the x-axis of the plot
        time = np.linspace(0, original_audio.duration_seconds, len(original_audio_data))

        # Create the plot of the original audio
        py_plt.figure(figsize=(10, 6))
        py_plt.plot(time, original_audio_data, label='Original Audio')
        py_plt.xlabel('Time (s)')
        py_plt.ylabel('Amplitude')
        py_plt.title('Original Audio with Repeating Phrase')
        py_plt.axvspan(repeating_phrase_start_time, repeating_phrase_end_time, color='red', alpha=0.3, label='Repeating Phrase')
        py_plt.legend()
        py_plt.show()

    def plot_audio_with_repeating_phrase_ICA(self, repeating_phrase_audio_file):
        # Load the original long audio file
        original_audio = self.audio

        # Load the shorter audio file containing the repeating phrase
        repeating_phrase_audio = AudioSegment.from_file(repeating_phrase_audio_file)

        # Resample the original audio to match the sample rate of the repeating phrase audio
        original_audio = original_audio.set_frame_rate(repeating_phrase_audio.frame_rate)

        # Extract the raw audio data from the original long audio file
        original_audio_data = np.array(original_audio.get_array_of_samples())

        # Extract the raw audio data from the repeating phrase audio file
        repeating_phrase_audio_data = np.array(repeating_phrase_audio.get_array_of_samples())

        # Ensure that both arrays have the same length along dimension 1
        min_len = min(len(original_audio_data), len(repeating_phrase_audio_data))
        original_audio_data = original_audio_data[:min_len]
        repeating_phrase_audio_data = repeating_phrase_audio_data[:min_len]

        # Perform ICA to separate the repeating phrase from the original audio
        ica = FastICA(n_components=2)
        S = np.vstack([original_audio_data, repeating_phrase_audio_data])
        S_ = ica.fit_transform(S.T).T
        repeating_phrase_separated = S_[1]

        # Convert time to seconds
        time = np.linspace(0, original_audio.duration_seconds, len(original_audio_data))

        # Create the plot of the original audio along with the separated repeating phrase
        py_plt.figure(figsize=(10, 6))
        py_plt.plot(time, original_audio_data, label='Original Audio')
        py_plt.plot(time, repeating_phrase_separated, label='Repeating Phrase (Separated)', color='red')
        py_plt.xlabel('Time (s)')
        py_plt.ylabel('Amplitude')
        py_plt.title('Original Audio with Separated Repeating Phrase (ICA)')
        py_plt.legend()
        py_plt.show()

    # def slice_segments(self):
    #     #map = runAssembly(self.file_name,cfg,self.file_path) #eventually were going to use this one
    #     file_name = '{}.json'.format(self.file_name[:-4])
    #     file_path = 
    #     map = loadDictFromJSON(file_name,self.file_path[:-len(self.file_name)])
    #     print(map)

    #########################
    # Graphing Functions
    def plot_dB(self, start: int, increment: int):

        # Extract the sample data as a numpy array
        signal_array = np.array(self.audio.get_array_of_samples())

        # Add a small constant value to avoid division by zero or taking the logarithm of zero
        signal_array = signal_array + 1e-9

        # Clip values to a small positive value to avoid negative values after adding the constant
        signal_array = np.clip(signal_array, 1e-9, None)

        # Converting the singal Array to dB
        signal_dB = 10*np.log10(signal_array)

        # Calculate time in milliseconds
        t_audio_ms = len(signal_array) * 1000 / self.audio.frame_rate

        # Generate time array
        times = np.linspace(0, t_audio_ms, num=len(signal_array))

        # Plot the signal wave
        py_plt.figure()
        py_plt.plot(times, signal_dB)
        py_plt.title(self.file_name[:-4])
        py_plt.ylabel('Amplitude (dB)')
        py_plt.xlabel('Time (ms)')
        py_plt.xlim(0, t_audio_ms)
        py_plt.ylim(0, 100)

        # Adding vertical lines
        py_plt.axvline(x=start, color='y')  # Starting point
        for i in range(50):
            py_plt.axvline(x=(start + ((i + 1) * increment)), color='r')  # Increment

        py_plt.show()

    def __plot_amp(self):

        # Extract the sample data as a numpy array
        signal_array = np.array(self.audio.get_array_of_samples())
        t_audio_ms = len(signal_array)/2 * 1000 / self.audio.frame_rate

        # # Debugging
        # print('Array Length: {}'.format(len(signal_array)))
        # print('Frame Rate: {}'.format(self.audio.frame_rate))
        # print('ms: {}'.format(t_audio_ms))

        # Generate time array
        times = np.linspace(0, t_audio_ms, num=len(signal_array))

        # Plot the signal wave
        py_plt.figure()
        py_plt.plot(times, signal_array)
        py_plt.title(self.file_name[:-4])
        py_plt.ylabel('Amplitude')
        py_plt.xlabel('Time (ms)')
        py_plt.xlim(0, t_audio_ms)

    def plot_amp(self, type=None | str, start=0, increment=0):
        self.__plot_amp()
            
        if type == 'inc':
            # Adding vertical lines
            py_plt.axvline(x=start, color='y')  # Starting point
            for i in range(50):
                py_plt.axvline(x=(start + ((i + 1) * increment)), color='r')  # Increment

        elif type == 'highlight':
            print('highlight')

        py_plt.show()

    #def plot_amp_regions():

    #########################
