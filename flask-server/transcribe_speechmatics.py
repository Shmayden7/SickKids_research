#########################
# Imports:
from analysis.json_helpers import dumpDictToJSON, parseOverwriteJSON
from analysis.ploting.ranking_normals import two_dimensional_grid
from analysis.ploting.plot_audio import plotWave
from analysis.audio_helpers import waveVAD
from analysis.word_analysis import segmentResponces
from analysis.audio_helpers import lowPassFilter

from api.assembly_ai.assembly_recorded import getAssemblyAIData
# from api.deepgram.deepgram_recorded_old import getDeepgramDataOld
# from api.deepgram.deepgram_recorded_new import getDeepgramDataNew
# from api.deepgram.deepgram_live import runDeepgramLiveTranscript

from api.assembly_ai.assembly_helpers import runAssembly
from api.deepgram.deepgram_helpers import parseRawDeepgram

# Classes
from classes.WavAudio import WavAudio

# Constants:
list = 1
condition_type = 'dLast'
eval_pbk = f'L{list}_Daniel_Dlast'

cfg = { 
    "punctuate": False,         # [Bool] Removes Puncuation from the transcription, Needs to be true for speaker_labels
    "format_text": False,       # [Bool] Returns the text as a simple string N.A. format
    "auto_highlights": False,   # [Bool] allows simpilar segmentation for 'say the word'
    "speaker_labels": False,    # [Bool] lets you know which person is speaking in a recording
    "speakers_expected": 1,     # [int] The Number of speakers Expected in a audio clip
    "use_boost": False,         # [Bool] If you want to use the suggested words
    "boost_param": 'high',      # [str: low default high] The Level at which we want to boost the ideal list
    "list_number": list,        # [int: 1 2] word list were analyzing for PBK
    # "audio_start_from": 0,      # [int] Start of analysis peroid (ms)
    # "audio_end_at": 280090,     # [int] End of analysis peroid (ms)
}
#########################
# Running:
# pbk = WavAudio(f'{eval_pbk}.wav',cfg,f'data/audio/pbk/normals/{condition}/')
pbk = WavAudio(f'{eval_pbk}.wav',cfg,f'data/audio/pbk/normals/deletion/L1_dLast/')
# pbk.plot_amp('inc', 4300, 4960)
# pbk.n_segments_fixed(4300, 4960, 50)
pbk.classify_cache_speechmatics(f'data/json/speechmatics/L{list}_{condition_type}/{eval_pbk}/')
#########################

#########################
### Deepgram AI
# deepgram_start = time.time()
# #asyncio.run(getDeepgramDataOld(eval_file, audio_path))
# deepgram_data = asyncio.run(getDeepgramDataNew(eval_file, audio_path))
# run_time = float("%.2f" % (time.time() - deepgram_start))
# deepgram_data.update({'run_time': run_time})

# print(json.dumps(deepgram_data, indent=3))
# print('Deepgram classification took {}s!'.format(run_time))

# dumpDictToJSON(deepgram_data, '{}.json'.format(eval_file[:-4]), dump_location=dump_location)

# asyncio.run(runDeepgramLiveTranscript())
#########################