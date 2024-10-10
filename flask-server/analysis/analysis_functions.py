#########################
# Imports:
import os
from data.constants import word_list

from analysis.json_helpers import loadDictFromJSON

# Constants:
#########################

##############################
# Description: Loads accuracy or confidence data from saved JSON files based on specified criteria.
# Parameters:
#   type_out (str): Specifies the type of data to load ('accuracy' or 'confidence').
#   list_number (int): Index to select the corresponding reference word.
#   json_path (str): Path to the directory containing the JSON files.
# Returns: 
#   list: A list of dictionaries containing folder names and their corresponding data.
##############################
def loadXFromSavedJson(type_out: str, list_number: int, json_path: str) -> dict:
    name = json_path.split('/')[-2]
    folders = []
    data_out = []

    # finding all the folder names in deletion
    for json_path, dirs, filenames in os.walk(json_path):
        folders.extend([{'name': x} for x in dirs])
        break
    folders = sorted(folders, key=lambda x: (x['name'].rsplit('_', 1)[-1] == 'Dfirst', x['name']))

    match type_out:

        case 'accuracy': # out struct contains accuracy values

            for folder in folders:
                path = json_path + folder['name'] +'/'

                data = []
                for index in range(50):

                    if name[3:] == 'normal':
                        file_name = '{}_{}'.format(index,folder['name'] + '.json')
                    elif name[3:] in ('deletion','dFirst','dLast'):
                        file_name = '{}_{}'.format(index,folder['name'].rsplit('_', 1)[0] + '.json')
                    elif name[3:] in ('c1000','c2000','c4000','c6000'):
                        file_name = '{}_{}'.format(index,folder['name'] + '.json')
                    
                    loaded_json = loadDictFromJSON(file_name, path)
                    loaded_json = loaded_json['results']

                    # normal case, transcription should match ref word
                    if name[3:] == 'normal':

                        if len(loaded_json) > 0:
                            content = loaded_json[0]['alternatives'][0]['content'].lower()
                            ref_word = word_list[list_number][index]

                            if content == ref_word:
                                data.append(1)
                            else:
                                data.append(0)
                        else:
                            data.append(0)

                    # deletion case, transcription should not match ref word
                    elif name[3:] == 'deletion'or'dFirst'or'dLast': 

                        if len(loaded_json) > 0:
                            content = loaded_json[0]['alternatives'][0]['content'].lower()
                            ref_word = word_list[list_number][index]

                            if content == ref_word:
                                data.append(0)
                            else:
                                data.append(1)
                        else: 
                            data.append(1)

                    # Low Pass case, transcription should not match ref word
                    elif name[3:] == 'c1000' or 'c2000' or 'c4000' or 'c6000': 

                        if len(loaded_json) > 0:
                            content = loaded_json[0]['alternatives'][0]['content'].lower()
                            ref_word = word_list[list_number][index]

                            if content == ref_word:
                                data.append(0)
                            else:
                                data.append(1)
                        else: 
                            data.append(1)

                data_out.extend([{'name': folder['name'], 'data': data}])

        case 'confidence':

            for folder in folders:
                path = json_path + folder['name'] +'/'

                data = []
                for index in range(50):

                    if name[3:] == 'normal':
                        file_name = '{}_{}'.format(index,folder['name'] + '.json')
                    elif name[3:] == 'deletion':
                        file_name = '{}_{}'.format(index,folder['name'].rsplit('_', 1)[0] + '.json')
                    elif name[3:] == 'c1000' or 'c2000' or 'c4000' or 'c6000':
                        file_name = '{}_{}'.format(index,folder['name'] + '.json')

                    loaded_json = loadDictFromJSON(file_name, path)
                    loaded_json = loaded_json['results']

                    if len(loaded_json) > 0:
                        confidence = loaded_json[0]['alternatives'][0]['confidence']
                        content = loaded_json[0]['alternatives'][0]['content'].lower()
                        ref_word = word_list[list_number][index]
                        
                        data.append(confidence)
                    else:
                        data.append(0)

                data_out.extend([{'name': folder['name'], 'data': data}])
    
    return data_out

##############################
# Description: Runs signal detection theory analysis by loading accuracy data and printing hit, miss, and rejection statistics.
# Parameters: None
# Returns: None
##############################
def run_sdt():
    L1_norm = loadXFromSavedJson('accuracy', 1, 'data/json/segmented_norms/L1_normal/')
    L2_norm = loadXFromSavedJson('accuracy', 2, 'data/json/segmented_norms/L2_normal/')
    norms = [L1_norm, L2_norm]

    L1_dele = loadXFromSavedJson('accuracy', 1, 'data/json/segmented_norms/L1_deletion/')
    L2_dele = loadXFromSavedJson('accuracy', 2, 'data/json/segmented_norms/L2_deletion/')
    dele = [L1_dele, L2_dele]

    L1_c1000 = loadXFromSavedJson('accuracy', 1, 'data/json/segmented_norms/L1_c1000/')
    L2_c1000 = loadXFromSavedJson('accuracy', 2, 'data/json/segmented_norms/L2_c1000/')

    L1_c2000 = loadXFromSavedJson('accuracy', 1, 'data/json/segmented_norms/L1_c2000/')
    L2_c2000 = loadXFromSavedJson('accuracy', 2, 'data/json/segmented_norms/L2_c2000/')

    L1_c4000 = loadXFromSavedJson('accuracy', 1, 'data/json/segmented_norms/L1_c4000/')
    L2_c4000 = loadXFromSavedJson('accuracy', 2, 'data/json/segmented_norms/L2_c4000/')

    L1_c6000 = loadXFromSavedJson('accuracy', 1, 'data/json/segmented_norms/L1_c6000/')
    L2_c6000 = loadXFromSavedJson('accuracy', 2, 'data/json/segmented_norms/L2_c6000/')
    lpf = [L1_c1000, L2_c1000, L1_c2000, L2_c2000, L1_c4000, L2_c4000, L1_c6000, L2_c6000]

    # Normal Data
    print('     Normals:')
    counter = 0
    for sublist in norms:
        hit = 0
        miss = 0
        for person in sublist:
            data = person['data']

            for x in data:
                if x == 1:
                    hit += 1
                elif x == 0:
                    miss += 1

        print(f'{counter}_hit: {hit}')
        print(f'{counter}_miss: {miss}')
        print(f'total: {miss+hit}')
        print()
        counter += 1

    # Deletions
    print('     Deletions:')
    counter = 0
    for sublist in dele:
        c_rej = 0
        f_ala = 0
        for person in sublist:
            data = person['data']

            for x in data:
                if x == 0:
                    c_rej += 1
                elif x == 1:
                    f_ala += 1

        print(f'{counter}_c_rej: {c_rej}')
        print(f'{counter}_f_ala: {f_ala}')
        print(f'total: {f_ala+c_rej}')
        print()
        counter += 1

    # LPF
    print('     LPF:')
    counter = 0
    for sublist in lpf:
        c_rej = 0
        f_ala = 0
        for person in sublist:
            data = person['data']

            for x in data:
                if x == 0:
                    c_rej += 1
                elif x == 1:
                    f_ala += 1

        print(f'{counter}_c_rej: {c_rej}')
        print(f'{counter}_f_ala: {f_ala}')
        print(f'total: {f_ala+c_rej}')
        print()
        counter += 1