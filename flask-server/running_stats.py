import pandas as pd
import statsmodels.api as sm
import json

from data.constants import word_list
from analysis.accuracy_plots import extract_data, read_json

word_list_1 = word_list[1]
word_list_2 = word_list[2]

master_lists = ['ursa.json', 'whisper large.json', 'whisper medium.json']
conditions = ['normal', 'dFirst', 'dLast', 'c1000', 'c2000', 'c4000', 'c6000']
people = ['aCauchi', 'Claire', 'Cole', 'Daniel', 'Hillary', 'Lulia', 'Melissa', 'Micheal', 'Polonenko', 'rCauchi', 'Robel', 'Samsoor']
dFirst_exceptions = ['are', 'is', 'end', 'own', 'axe', 'all', 'as', 'on']
dLast_exceptions = ['no', 'tree', 'lay', 'me', 'few', 'plow', 'gray', 'bee', 'grew', 'knee', 'tray']

data = {}
data_long = []
for model in master_lists:
    model_data = []
    for condition in conditions:

        L1_data, L2_data = extract_data(model, condition)

        for speaker in people:
            spoken_words_L1 = L1_data[speaker]
            spoken_words_L2 = L2_data[speaker]

            for i, spoken_word in enumerate(spoken_words_L1):
                # Help me with this if statemtnt
                if (condition == 'dFirst' and word_list_1[i] in dFirst_exceptions) or (condition == 'dLast' and word_list_1[i] in dLast_exceptions):
                    continue  # Skip this iteration instead of breaking out of the loop

                dummy = {
                    "listener": model.split('.json')[0],
                    "condition": condition,
                    "speaker": speaker,
                    "word": word_list_1[i],
                }

                if spoken_word == word_list_1[i]:
                    dummy['accuracy'] = 1
                else:
                    dummy['accuracy'] = 0

                model_data.append(dummy)
                data_long.append(dummy)

            for i, spoken_word in enumerate(spoken_words_L2):
                if (condition == 'dFirst' and word_list_2[i] in dFirst_exceptions) or (condition == 'dLast' and word_list_2[i] in dLast_exceptions):
                    continue  # Skip this iteration instead of breaking out of the loop

                dummy = {
                    "listener": model.split('.json')[0], 
                    "condition": condition,
                    "speaker": speaker,
                    "word": word_list_2[i],
                }

                if spoken_word == word_list_2[i]:
                    dummy['accuracy'] = 1
                else:
                    dummy['accuracy'] = 0

                model_data.append(dummy)
                data_long.append(dummy)


    data[model] = model_data

with open('data/inter_rater_assess.json', 'r') as file:
        inter_rater_data = json.load(file)

human_assess_long = []
human_assess_indi = []
human_raters = ['cole', 'joyce', 'ryley', 'lulia', 'isabel', 'susan', 'maria']
for row in inter_rater_data:
    condition = row['condition']
    if condition == 'Normal':
        condition = 'normal'
    speaker = row['speaker']
    target_word = row['word']

    for evaluator in human_raters:

        if (condition == 'dFirst' and row['word'] in dFirst_exceptions) or (condition == 'dLast' and row['word'] in dLast_exceptions):
            continue  # Skip this iteration instead of breaking out of the loop

        if target_word == row['responses'][evaluator]:
            accuracy = 1
        else:
            accuracy = 0

        dummy_human_assess = {
                    "listener": evaluator, 
                    "condition": condition,
                    "speaker": speaker,
                    "word": target_word,
                    "accuracy": accuracy,
                }
        
        dummy_long = {
                "listener": 'Human', 
                "condition": condition,
                "speaker": speaker,
                "word": target_word,
                "accuracy": accuracy,
            }
        
        data_long.append(dummy_long)
        human_assess_long.append(dummy_long)
        human_assess_indi.append(dummy_human_assess)

data['Human'] = human_assess_long

df_long = pd.DataFrame(data_long)
df_long.to_excel('/Users/ayden/Desktop/SickKids/excel/data_long/all classifiers.xlsx', index=False)

df_human = pd.DataFrame(human_assess_indi)
df_human.to_excel('/Users/ayden/Desktop/SickKids/excel/data_long/human with evals.xlsx', index=False)

for model in data:

    model_data = data[model]
    df = pd.DataFrame(model_data)

    # Save the DataFrame to an Excel file
    output_file = f"/Users/ayden/Desktop/SickKids/excel/data_long/{model.replace('.json', '')}_data.xlsx"
    df.to_excel(output_file, index=False)