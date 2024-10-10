#########################
# Imports:
import json
import string
import os
import re
from data.constants import word_list

# import openpyxl

# Constants:
#########################

##############################
# Description: Loads a dictionary from a JSON file.
# Parameters: 
#   - data (str): The name of the JSON file.
#   - dir (str): The local directory containing the file.
# Returns: None
# Dependencies: None
##############################
def loadDictFromJSON(file_name: str, dir='data/json/') -> dict:
    try:
        with open(dir+file_name) as loaded_file:
            return json.load(loaded_file)
    except IOError as e:
        print('Error, could not open {}'.format(file_name))
        print(e)
        exit()

##############################
# Description: Loads a dictionary from a JSON file.
# Parameters: 
#   - data (str): The name of the JSON file.
#   - dir (str): The local directory containing the file.
# Returns: None
# Dependencies: None
##############################
def dumpDictToJSON(data, file_name: str, dump_location='data/json/'):
    try:
        with open((dump_location+file_name), 'w') as a:
            json.dump(data, a, indent=3)
        print('{} dumped to JSON'.format(file_name))
    except IOError as e:
        print('Error, could not open {}'.format(file_name))
        print(e)
        exit()

##############################
# Description: Removes specified words from the "text" and "words" fields of a JSON object.
# Parameters: 
#   - file_name (str): The name of the JSON file.
#   - parsed (list): The list of words to be removed.
#   - location (str): The local directory of the JSON file.
# Returns: Overwrites the JSON file at the same location.
# Dependencies: loadDictFromJSON(), dumpDictToJSON()
##############################
def parseOverwriteJSON(file_name: str, parsed: str, dir='data/json/'):
    dict_pre = loadDictFromJSON(file_name, dir)
    dict_post = {
        "list_number": dict_pre['list_number'],
        "audio_duration": dict_pre['audio_duration'],
        "run_time": dict_pre['run_time'],
    }
    removing_words = parsed.split() 
    returned_words_dict = []
    for word in dict_pre['words']:
        word_string = word['text']
        word_string = word_string.translate(str.maketrans('','', '.?!')) # removes peroids
        word_string = word_string.lower() # converts capitalized to lowercase
        word['text'] = word_string
        if not (word['text'] in removing_words):
            returned_words_dict.append(word)
    dict_post.update({'words': returned_words_dict})

    updated_filename = file_name[:-5] + '_parsed.json'
    dumpDictToJSON(dict_post, updated_filename, dir)

#########################
# Desc: Removes input words from the text and words field of a JSON Object
# Params: file_name: name of the JSON file, parsed: words we want to remove, Location: local dir
# Return: Overwrites the JSON file in same location
# Dependant on: loadDictFromJSON(), dumpDictToJSON()
#########################

##############################
# Description: Removes specified words from the "text" and "words" fields of a JSON object.
# Parameters: 
#   - file_name (str): The name of the JSON file.
#   - parsed (list): The list of words to be removed.
#   - location (str): The local directory of the JSON file.
# Returns: Overwrites the JSON file in the same location.
# Dependencies: loadDictFromJSON(), dumpDictToJSON()
##############################
def intraterRel():
    temp_location = '/Users/ayden/Desktop/paper/temp.xlsx'
    
    workbook = openpyxl.load_workbook(temp_location)
    sheet = workbook.active
    transcriptions = []

    for row_number, row in enumerate(sheet.iter_rows(min_row=2, values_only=True), start=2):
        if row_number > 351: break

        row = row[:4]
        print(row)

        person = str(row[2])
        condition = str(row[1])
        if row[3] <= 49:
            word = row[3]
            list = 1
        else:
            word = row[3]-50
            list = 2

        condition_dict = {
            '1': 'c1000',
            '2': 'c2000',
            '4': 'c4000',
            '6': 'c6000',
            'N': 'normal',
            'F': 'dFirst',
            'L': 'dLast',
        }

        person_dict = {
            '1': 'aCauchi',
            '2': 'Claire',
            '3': 'Cole',
            '4': 'Daniel',
            '5': 'Hillary',
            '6': 'Lulia',
            '7': 'Melissa',
            '8': 'Micheal',
            '9': 'Polonenko',
            '10': 'rCauchi',
            '11': 'Robel',
            '12': 'Samsoor',
        }

        if condition == 'F' or condition == 'L':
            print('deletion')
            end_bit = ''
            if condition == 'F':
                end_bit = 'dFirst'
            elif condition == 'L':
                end_bit = 'dLast'

            path_string = f'data/json/segmented_norms/L{list}_{condition_dict[condition]}/L{list}_{person_dict[person]}_{end_bit}/'
            print(path_string)
        else:
            print('non-deletion')
            path_string = f'data/json/segmented_norms/L{list}_{condition_dict[condition]}/L{list}_{person_dict[person]}/'
            print(path_string)

        obj = loadDictFromJSON(f'{word}_L{list}_{person_dict[person]}.json', path_string)
        if len(obj['results']) > 0:
            transcription = obj['results'][0]['alternatives'][0]['content']
        else:
            transcription = ''

        transcriptions.append(transcription.lower())
    print(transcriptions)

    for i, value in enumerate(transcriptions):
        sheet.cell(row=2 + i, column=6, value=value)

    workbook.save('/Users/ayden/Desktop/paper/temp_updated.xlsx')
    workbook.close()

##############################
# Description: Pulls D-prime metrics for a given person based on normal, frequency,
#              and deletion shift data.
# Parameters:
#   - person (str): The name of the person whose data is being processed.
# Returns: A dictionary containing hit, miss counts, and other related data.
# Dependencies: loadDictFromJSON()
##############################
def pullDPrimeMetricsByPerson(person: str) -> dict:
    hit = miss = 0
    other_data = {
        'c_rej': {
            'dFirst': 0,
            'dLast': 0,
            'c1000': 0,
            'c2000': 0,
            'c4000': 0,
            'c6000': 0,
        },
        'f_alarm': {
            'dFirst': 0,
            'dLast': 0,
            'c1000': 0,
            'c2000': 0,
            'c4000': 0,
            'c6000': 0,
        }
    }

    frequency = ['c1000','c2000','c4000','c6000']
    deletion = ['dFirst','dLast']

    # Pulling normal data
    for i in range(1,3):
        normals_path = f'data/json/segmented_norms/L{i}_normal/L{i}_{person}/'
        
        for x in range(50): 
            fetched_json = loadDictFromJSON(f'{x}_L{i}_{person}.json',normals_path)
            fetched_json = fetched_json['results']

            if len(fetched_json) > 0:
                content = fetched_json[0]['alternatives'][0]['content'].lower()
                ref_word = word_list[i][x]

                if content == ref_word:
                    hit = hit + 1
                else:
                    miss = miss + 1
            else:
                miss = miss + 1

    # Pulling frequency shift data
    for i in range(1,3):
        for freq in frequency:
            freq_path = f'data/json/segmented_norms/L{i}_{freq}/L{i}_{person}/'

            for x in range(50):
                fetched_json = loadDictFromJSON(f'{x}_L{i}_{person}.json',freq_path)
                fetched_json = fetched_json['results']

                if len(fetched_json) > 0:
                    content = fetched_json[0]['alternatives'][0]['content'].lower()
                    ref_word = word_list[i][x]

                    if content == ref_word:
                        other_data['f_alarm'][freq] = other_data['f_alarm'][freq] + 1
                    else:
                        other_data['c_rej'][freq] = other_data['c_rej'][freq] + 1
                else:
                    other_data['c_rej'][freq] = other_data['c_rej'][freq] + 1

    # Pulling deletion shift data
    for i in range(1,3):
        for dele in deletion:
            del_path = f'data/json/segmented_norms/L{i}_{dele}/L{i}_{person}_{dele}/'

            for x in range(50):
                fetched_json = loadDictFromJSON(f'{x}_L{i}_{person}.json',del_path)
                fetched_json = fetched_json['results']

                if len(fetched_json) > 0:
                    content = fetched_json[0]['alternatives'][0]['content'].lower()
                    ref_word = word_list[i][x]

                    if content == ref_word:
                        other_data['f_alarm'][dele] = other_data['f_alarm'][dele] + 1
                    else:
                        other_data['c_rej'][dele] = other_data['c_rej'][dele] + 1
                else:
                    other_data['c_rej'][dele] = other_data['c_rej'][dele] + 1
        
    out = {
        "hit": hit,
        "miss": miss,
        'other_data': other_data,
    }

    return out

##############################
# Description: Processes JSON files in a directory by modifying the 'text' field
#              to remove spaces, punctuation, and convert it to lowercase.
# Parameters: 
#   - json_directory (str): The directory containing the JSON files.
# Returns: None. Prints the filename and modified text for each file.
# Dependencies: os, re, json
##############################
def process_whisper_json(json_directory):
    # Function to extract the numerical value before the first underscore
    def extract_number(filename):
        match = re.match(r"(\d+)_", filename)
        return int(match.group(1)) if match else float('inf')

    # Get and sort the list of JSON files in the directory based on the extracted number
    json_files = sorted(
        [f for f in os.listdir(json_directory) if f.endswith(".json")],
        key=extract_number
    )

    processed_files_count = 0
    for filename in json_files:
        json_path = os.path.join(json_directory, filename)
        
        # Read the JSON file
        with open(json_path, "r") as json_file:
            data = json.load(json_file)
        
        # Modify the 'text' field
        if 'text' in data:
            original_text = data['text']
            modified_text = original_text.replace(" ", "").replace(".", "").replace(",", "").replace("!", "").replace("?", "").lower()
            data['text'] = modified_text
        
        with open(json_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

        print(f"Processed {filename}: {data['text']}")
        
        processed_files_count += 1

    print(f"Total number of files processed: {processed_files_count}")

##############################
# Description: Preprocesses text by converting it to lowercase and removing punctuation.
# Parameters: 
#   - content (str): The input text to be processed.
# Returns: The processed text, or None if input is empty.
# Dependencies: string
##############################
def preprocess_content(content):
    if content:
        content = content.lower()
        content = content.translate(str.maketrans('', '', string.punctuation))
    return content

##############################
# Description: Retrieves a sorted list of subdirectories within a given base directory.
# Parameters: 
#   - base_dir (str): The directory containing the subdirectories to be listed.
# Returns: A list of sorted subdirectory paths.
# Dependencies: os
##############################
def get_sorted_subdirs(base_dir):
    subdirs = [os.path.join(base_dir, d) for d in sorted(os.listdir(base_dir)) if os.path.isdir(os.path.join(base_dir, d))]
    return subdirs

##############################
# Description: Extracts the numerical value preceding the first underscore in a filename.
# Parameters: 
#   - file_name (str): The name of the file from which to extract the number.
# Returns: The extracted number as an integer, or infinity if no match is found.
# Dependencies: re
##############################
def extract_number(file_name):
    match = re.match(r"(\d+)_", file_name)
    return int(match.group(1)) if match else float('inf')

##############################
# Description: Retrieves and sorts JSON files in a specified subdirectory based on 
#              the numerical value before the first underscore in the filename.
# Parameters: 
#   - subdir (str): The subdirectory containing the JSON files.
# Returns: A list of sorted JSON file paths.
# Dependencies: os, extract_number
##############################
def get_sorted_json_files(subdir):
    json_files = [f for f in os.listdir(subdir) if f.endswith('.json')]
    json_files.sort(key=extract_number)
    return [os.path.join(subdir, f) for f in json_files]

##############################
# Description: Combines 'content' data from JSON files in sorted subdirectories into a 
#              single output file, ensuring exactly 50 entries per subdirectory.
# Parameters: 
#   - input_base_dir (str): The base directory containing subdirectories with JSON files.
#   - final_output_file (str): The path to the output JSON file for combined data.
# Returns: None. Outputs a combined JSON file at the specified location.
# Dependencies: get_sorted_subdirs(), get_sorted_json_files(), preprocess_content()
##############################
def concat_content_json_speechmatics(input_base_dir, final_output_file):
    
    final_combined_data = {}
    sorted_subdirs = get_sorted_subdirs(input_base_dir)
    
    # Read and combine the 'content' data from each subdirectory
    for subdir in sorted_subdirs:
        deepest_dir_name = os.path.basename(subdir)
        
        sorted_json_files = get_sorted_json_files(subdir)
        combined_content = []
        
        for json_file_path in sorted_json_files:
            with open(json_file_path, 'r') as json_file:
                print(json_file)  # Debugging: Print the JSON file being processed
                json_data = json.load(json_file)
                
                # Extract the 'content' field, preprocess, and append to the combined content list
                if json_data['results']:
                    for result in json_data['results']:
                        content = result['alternatives'][0].get('content', '')
                        processed_content = preprocess_content(content)
                        combined_content.append(processed_content)
                else:
                    combined_content.append('')
        
        # Ensure there are exactly 50 strings for this subdirectory, filling with empty strings if necessary
        while len(combined_content) < 50:
            combined_content.append('')
        
        name_key = deepest_dir_name  # Maintain L1_ and L2_ prefixes
        final_combined_data[name_key] = combined_content[:50]  # Ensure only 50 strings
    
    # Write the final combined data to the specified final output file
    with open(final_output_file, 'w') as final_output:
        json.dump(final_combined_data, final_output, indent=4)
    
    print(f"Final combined JSON file written to: {final_output_file}")

##############################
# Description: Combines 'text' data from JSON files in sorted subdirectories into a 
#              single output file, ensuring exactly 50 entries per subdirectory.
# Parameters: 
#   - input_base_dir (str): The base directory containing subdirectories with JSON files.
#   - final_output_file (str): The path to the output JSON file for combined data.
# Returns: None. Outputs a combined JSON file at the specified location.
# Dependencies: get_sorted_subdirs(), get_sorted_json_files(), preprocess_content()
##############################
def concat_content_json_whisper(input_base_dir, final_output_file):
    
    final_combined_data = {}
    sorted_subdirs = get_sorted_subdirs(input_base_dir)
    
    # Read and combine the 'text' data from each subdirectory
    for subdir in sorted_subdirs:
        deepest_dir_name = os.path.basename(subdir)
        
        sorted_json_files = get_sorted_json_files(subdir)
        combined_content = []
        
        # Read and combine the 'text' data from each file in the subdirectory
        for json_file_path in sorted_json_files:
            # Read the JSON file content
            with open(json_file_path, 'r') as json_file:
                json_data = json.load(json_file)
                
                content = json_data.get('text', '')
                processed_content = preprocess_content(content)
                combined_content.append(processed_content)
        
        # Ensure there are exactly 50 strings for this subdirectory, filling with empty strings if necessary
        while len(combined_content) < 50:
            combined_content.append('')
        
        name_key = deepest_dir_name  # Maintain L1_ and L2_ prefixes
        final_combined_data[name_key] = combined_content[:50]  # Ensure only 50 strings
    
    with open(final_output_file, 'w') as final_output:
        json.dump(final_combined_data, final_output, indent=4)
    
    print(f"Final combined JSON file written to: {final_output_file}")