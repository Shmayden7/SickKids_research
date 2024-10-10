#########################
# Imports:
import time, json

from api.assembly_ai.assembly_recorded import getAssemblyAIData
from analysis.json_helpers import dumpDictToJSON

# Constants:
#########################

#########################
# Desc: Removes un-necessary information from the returned api dict
# Params: raw data dict, single string of attributes, space separated
# Return: dict, Assembly AI data that we care about
# Dependant on: N.A.
#########################
def parseRawAssembly(data: dict, attributes: str):

    # splits up the single string by spaces into array of attributes
    attribute_array = attributes.split()

    important_data = {} # the attributes we acc want

    for i in range(len(attribute_array)):
        print(attribute_array[i])
        string_1 = attribute_array[i]
        string_2 = data[attribute_array[i]]
        new_field = {string_1: string_2}
        print(new_field)
        important_data.update(new_field)

    return important_data

#########################
# Desc: saves the response as a text file
# Params: data fetched from the API
# Return: N.A., saves a txt file to same directory
# Dependant on: N.A.
#########################
def saveTranscriptToTxt(data, file_name='Assembly_AI_Transcription'):
    if data:
        text_file_name =  file_name + '.txt'
        with open(text_file_name, 'w') as f:
            f.write(data['text'])
        print('Transcription complete!!!')

#########################
# Desc: Runs Assembly Ai Poll, uploads an MP3/wav, saves the responce as a json
# Params: data fetched from the API
# Return: N.A., saves a txt file to same directory
# Dependant on: N.A.
#########################
def runAssembly(file_name: str, cfg: dict, audio_path='data/audio/', dump_location='data/json/'):
    print('Running Assembly!')
    timer_start = time.time()
    data = getAssemblyAIData(file_name, cfg, audio_path)
    run_time = float("%.2f" % (time.time() - timer_start))
    data.update({'run_time': run_time})
    data.update({'list_number': cfg['list_number']})
            
    print('Assembly Classification Complete! Runtime: {}s'.format(run_time))
    dumpDictToJSON(data, '{}.json'.format(file_name[:-4]), dump_location=dump_location)
    print()