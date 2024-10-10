#########################
# Imports:
from analysis.ploting.plot_audio import plotWave
from analysis.analysis_functions import run_sdt
from analysis.analysis_functions import loadXFromSavedJson
from analysis.audio_helpers import segment_pbk_recording
from analysis.json_helpers import intraterRel, concat_content_json_speechmatics, concat_content_json_whisper
from analysis.excel import excel, percent_agreement
from analysis.ML_class import dumb_python
from analysis.json_helpers import loadDictFromJSON, pullDPrimeMetricsByPerson

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy.stats import norm, beta
import seaborn as sns

import pandas as pd
import json
import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa

list_number = 2
kind = 'deletion'
#########################
# hit  = 46
# miss = 4

# altered_data = [
#     [35,15],
#     [28,22],
#     [43,7],
#     [24,26],
#     [7,43],
#     [9,41],
# ]

# values = []
# for i in range(len(altered_data)):
#     number = dprime(hit,miss,altered_data[i][0],altered_data[i][1])
#     values.append(round(number,2))
# print(values)

# Read the Excel file
# file_path = '/Users/ayden/Desktop/Book1.xlsx'
# df = pd.read_excel(file_path)

# # Transform the data
# transformed_data = pd.DataFrame({
#     'Yes': df.sum(axis=1),
#     'No': df.shape[1] - df.sum(axis=1)
# })

# # Calculate Fleiss' kappa
# kappa_value = fleiss_kappa(transformed_data.to_numpy())
# print(f'Fleiss\' kappa: {kappa_value}')
#########################
# file_name = 'L1_Daniel.wav'
# start = 1000
# incroment = 5000
# audio_path = 'data/audio/pbk/normals/base/'
# output_path = 'data/audio/pbk/normals/base/L1_Daniel'


# plotWave(file_name,start,incroment,audio_path)
# segment_pbk_recording(file_name,start,incroment,50,audio_path,output_path)

list_number = 2
condition = 'normal'
# concat_content_json_speechmatics(f'data/json/speechmatics/L{list_number}_{condition}', f'data/json/master/speechmatics_L{list_number}_{condition}.json')
concat_content_json_whisper(f'data/json/deepspeech/L{list_number}_{condition}', f'data/json/master/deepspeech_L{list_number}_{condition}.json')

### Checking The Length ###
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def print_array_lengths(data):
    for name, array in data.items():
        if isinstance(array, list):
            print(f'{name}: {len(array)}')
        else:
            print(f'{name} does not have a list associated with it.')

file_path = f'data/json/master/deepspeech_L{list_number}_{condition}.json'

data = read_json_file(file_path)
print_array_lengths(data)
###########################