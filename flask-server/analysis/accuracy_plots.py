#########################
# Imports:
import json
import os
import matplotlib.pyplot as plt
import statsmodels.stats.inter_rater as ir
import numpy as np
from scipy.stats import norm
from scipy.stats import gaussian_kde
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import to_rgba
from collections import Counter, defaultdict
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
from matplotlib.legend_handler import HandlerTuple
import pandas as pd
from pandas.plotting import table
from data.constants import word_list
#########################

##############################
# Description: Reads a JSON file from a specified path and returns its contents.
# Parameters: file_name (str): The name of the JSON file to read.
# Returns: dict: The data loaded from the JSON file.
##############################
def read_json(file_name: str) -> dict:
    base_path = 'data/json/master'
    file_path = os.path.join(base_path, file_name)

    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

##############################
# Description: Extracts L1 and L2 data for a specified condition from a JSON file.
# Parameters: 
#   - master_name (str): The name of the JSON file to read.
#   - condition (str): The condition to filter data.
# Returns: 
#   - tuple: L1 and L2 data dictionaries for the specified condition.
##############################
def extract_data(master_name: str, condition: str):
    data = read_json(master_name)
    L1_data = data.get(f'L1_{condition}', {})
    L2_data = data.get(f'L2_{condition}', {})
    
    return L1_data, L2_data

##############################
# Description: Compares two word lists and scores matches; returns a list of scores.
# Parameters: 
#   - ideal_list (list): The ideal list of words.
#   - eval_list (list): The evaluated list of words.
# Returns: 
#   - list: A list of scores (1 for match, 0 for no match).
##############################
def compare_word_lists(ideal_list:list, eval_list:list) -> list:
    # Ensure the lists are of the same length
    assert len(ideal_list) == len(eval_list), "Lists must be of the same length"
    
    results = []

    for ideal_word, eval_word in zip(ideal_list, eval_list):
        # This is where scoring is performed
        if ideal_word == eval_word:
            results.append(1)
        else:
            results.append(0)

    return results

##############################
# Description: Scores evaluation results for two conditions by each person; returns a dictionary of scores.
# Parameters:
#   - L1_data (dict): Evaluation data for the first condition.
#   - L2_data (dict): Evaluation data for the second condition.
#   - people (list): List of individuals to evaluate.
# Returns:
#   - dict: A dictionary mapping each person to their combined evaluation scores.
##############################
def score_condition_by_person(L1_data:dict, L2_data:dict, people:list) -> dict:
    scores_by_person = {}

    L1_ideal = word_list[1]
    L2_ideal = word_list[2]

    for person in people:
        L1_eval = L1_data.get(person, {})
        L2_eval = L2_data.get(person, {})

        # print(f'Length of {person} L1: {len(L1_eval)}')
        # print(f'Length of {person} L2: {len(L2_eval)}\n')

        L1_results = compare_word_lists(ideal_list=L1_ideal, eval_list=L1_eval)
        L2_results = compare_word_lists(ideal_list=L2_ideal, eval_list=L2_eval)

        total_results = L1_results + L2_results

        scores_by_person[person] = total_results
        
    return scores_by_person

##############################
# Description: Calculates the average scores across individuals for each evaluation position.
# Parameters:
#   - scores_by_person (dict): A dictionary mapping each person to their evaluation scores.
# Returns:
#   - list: A list of average scores for each evaluation position, rounded to two decimal places.
##############################
def accuracy_over_people(scores_by_person: dict) -> list:
    # Get the number of people
    num_people = len(scores_by_person)
    assert num_people > 0, "The Number of People must be > 0 to Perform an Average"

    # Initialize an array to store the sum of scores and count of non-None values at each position
    score_length = len(next(iter(scores_by_person.values())))
    sum_scores = [0] * score_length
    count_scores = [0] * score_length

    # Iterate over each person's scores
    for scores in scores_by_person.values():
        for i, score in enumerate(scores):
            if score is not None:
                sum_scores[i] += score
                count_scores[i] += 1

    # Calculate the average at each position and round to two decimal places
    average_scores = [round(sum_score / count, 2) if count > 0 else None for sum_score, count in zip(sum_scores, count_scores)]
    return average_scores

##############################
# Description: Computes the standard error of scores across individuals for each evaluation position.
# Parameters:
#   - scores_by_person (dict): A dictionary mapping each person to their evaluation scores.
# Returns:
#   - list: A list of standard errors for each evaluation position, rounded to two decimal places.
##############################
def std_error_over_people(scores_by_person: dict) -> list:
    # Get the number of people
    num_people = len(scores_by_person)
    assert num_people > 0, "The Number of People must be > 0 to Perform the Calculation"

    # Initialize an array to store all scores at each position
    score_length = len(next(iter(scores_by_person.values())))
    all_scores = np.zeros((num_people, score_length))

    # Collecting scores
    for i, scores in enumerate(scores_by_person.values()):
        all_scores[i] = scores

    std_devs = np.std(all_scores, axis=0, ddof=1)
    standard_errors = std_devs / np.sqrt(num_people)
    standard_errors = [round(se, 2) for se in standard_errors]

    return standard_errors

##############################
# Description: Maps words to their accuracy and standard error scores, excluding exceptions
# Parameters:
#   - accuracy_scores_by_word (list): Accuracy scores for words.
#   - standard_errors_by_word (list): Standard errors for the accuracy scores.
#   - condition (str): Condition determining which exceptions to exclude.
# Returns:
#   - dict: Dictionary with words as keys and (accuracy, standard error) tuples as values.
##############################
def create_word_acc_std(accuracy_scores_by_word: list, standard_errors_by_word: list, condition: str) -> dict:
    assert len(accuracy_scores_by_word) == 100, "There Must Be 100 Accuracy Scores Available"
    assert len(standard_errors_by_word) == 100, "There Must Be 100 Standard Errors Available"
    
    word_acc_std_dict = {}

    total_word_list = word_list[1] + word_list[2]

    # Only include words with non-None accuracy values
    for acc, err, word in zip(accuracy_scores_by_word, standard_errors_by_word, total_word_list):
        if acc is not None:
            word_acc_std_dict[word] = (acc, err)

    # Define exceptions based on condition
    if condition == 'dFirst':
        exceptions = ['are', 'is', 'end', 'own', 'axe', 'all', 'as', 'on']
    elif condition == 'dLast':
        exceptions = ['no', 'tree', 'lay', 'me', 'few', 'plow', 'gray', 'bee', 'grew', 'knee', 'tray']
    else:
        exceptions = []

    # Remove exceptions from word_acc_std_dict
    word_acc_std_dict = {word: val for word, val in word_acc_std_dict.items() if word not in exceptions}

    # Sorting according to accuracy
    word_acc_std_dict = dict(sorted(word_acc_std_dict.items(), key=lambda item: item[1][0], reverse=True))

    # print(word_acc_std_dict)
    # print(f"Length of word_acc_std_dict: {len(word_acc_std_dict)}")

    return word_acc_std_dict

##############################
# Description: Nullifies scores for specified words in the input data dictionary.
# Parameters:
#   - data (dict): A dictionary with identifiers as keys and score lists as values.
#   - words_removed (list): Words for which scores should be set to None.
# Returns:
#   - dict: Modified dictionary with nullified scores.
##############################
def modify_scores(data: dict, words_removed: list) -> dict:

    list1 = word_list[1]
    list2 = word_list[2]
    
    # Find the indices of the words in the input word list
    indices_to_nullify = [i for i, word in enumerate(list1) if word in words_removed]
    indices_to_nullify += [i for i, word in enumerate(list2) if word in words_removed]
    
    # Set the corresponding values in the c1_scores dictionary to None
    for key in data:
        for idx in indices_to_nullify:
            if idx < len(data[key]):
                data[key][idx] = None
    
    return data

##############################
# Description: Reorganizes inter-rater assessment data into a structured format 
#              by condition and rater, with grades based on ideal words.
# Returns:
#   - dict: A dictionary with conditions as keys and dictionaries of raters and their 
#            corresponding grades as values.
##############################
def re_order_interrater() -> dict:
    # Load the existing JSON data
    with open('data/inter_rater_assess.json', 'r') as file:
        data = json.load(file)

    # Initialize the new structure
    new_data = {}

    conditions = ['Normal', 'dFirst', 'dLast', 'c1000', 'c2000', 'c4000', 'c6000']
    raters = ['ursa', 'cole', 'joyce', 'ryley', 'lulia', 'isabel', 'susan', 'maria']

    for condition in conditions:
        data_subset = [entry for entry in data if entry.get('condition') == condition]
        
        for entry in data_subset:
            entry.pop('speaker', None)
            entry.pop('word number', None)
            entry.pop('condition', None)

        temp_condition = {}

        for rater in raters:
            raters_grades = []

            for field in data_subset:
                ideal_word = field.get('word')
                raters_response = field.get('responses', {}).get(rater)

                # Debug print statements
                if ideal_word is None or raters_response is None:
                    print(f"Missing data in entry: {field}")
                    continue

                if ideal_word == raters_response:
                    grade = 1 
                else:
                    grade = 0

                raters_grades.append(grade)

            temp_condition[rater] = raters_grades
        
        new_data[condition] = temp_condition

    return new_data        


    # # Iterate through the entries in the data
    # for entry in data:
    #     condition = entry['condition']
    #     speaker = entry['speaker']
    #     word_number = entry['word_number']
    #     word = entry['word']

    #     # Initialize the condition and speaker if not already present
    #     if condition not in new_data:
    #         new_data[condition] = {}
    #     if speaker not in new_data[condition]:
    #         new_data[condition][speaker] = []

    #     # Append the grade to the speaker's list for the condition
    #     new_data[condition][speaker].append(word_number)

    # return new_data

##############################
# Description: Calculates true positives, false positives, false negatives, and true negatives 
#              based on the evaluation results in the input data.
# Parameters:
#   - positive (bool): Indicates whether to evaluate for positive class.
#   - data (dict): A dictionary where keys are identifiers and values are lists of evaluation results.
# Returns:
#   - dict: A dictionary containing counts of TP, FP, FN, and TN.
##############################
def determine_P_N(positive: bool, data: dict) -> dict:
    total_values = {
        "TP": 0,
        "FP": 0,
        "FN": 0,
        "TN": 0
    }

    for key, values in data.items():
        for value in values:
            if positive:
                if value == 1:
                    total_values["TP"] += 1
                else:
                    total_values["FN"] += 1
            else:
                if value == 1:
                    total_values["FP"] += 1
                else:
                    total_values["TN"] += 1

    return total_values

##############################
# Description: Computes accuracy percentage from true/false positives and negatives
# Parameters:
#   - TP (int): True positives count.
#   - TN (int): True negatives count.
#   - FP (int): False positives count.
#   - FN (int): False negatives count.
# Returns:
#   - float: The accuracy percentage rounded to two decimal places.
##############################
def determine_accuracy(TP:int,TN:int,FP:int,FN:int) -> float:
    acc = (TP+TN)/(TP+TN+FP+FN)
    return round(acc*100,2)

##############################
# Description: Computes precision percentage from true and false positives.
# Parameters:
#   - TP (int): True positives count.
#   - FP (int): False positives count.
# Returns:
#   - float: The precision percentage rounded to two decimal places.
##############################
def determine_precision(TP:int, FP:int):
    if (TP + FP) == 0:
        return 0  # Prevent division by zero
    precision = TP / (TP + FP)
    return round(precision * 100, 2)

##############################
# Description: Computes recall percentage from true positives and false negatives.
# Parameters:
#   - TP (int): True positives count.
#   - FN (int): False negatives count.
# Returns:
#   - float: The recall percentage rounded to two decimal places.
##############################
def determine_recall(TP:int, FN:int) -> float:
    if (TP + FN) == 0:
        return 0  # Prevent division by zero
    recall = TP / (TP + FN)
    return round(recall * 100, 2)

##############################
# Description: Computes the F1 score based on precision and recall values.
# Parameters:
#   - precision (float): The precision percentage.
#   - recall (float): The recall percentage.
# Returns:
#   - float: The F1 score rounded to two decimal places.
##############################
def determine_fscore(precision:float, recall:float) -> float:
    if (precision + recall) == 0:
        return 0  # Prevent division by zero
    fscore = 2 * (precision * recall) / (precision + recall)
    return round(fscore, 2)

##############################
# Description: Creates accuracy plots with scatter and kernel density for PBK word responses.
# Parameters:
#   - data (dict): A dictionary mapping words to tuples of (accuracy, standard error).
# Returns:
#   - list: Words with accuracy above 80% after plotting.
##############################
def create_acc_plot_unaltered(data: dict) -> list:
    # Sort the dictionary by accuracy values in descending order
    sorted_data = sorted(data.items(), key=lambda item: item[1][0], reverse=True)

    # Extract words, accuracies, and standard errors
    words, values = zip(*sorted_data)
    accuracies, standard_errors = zip(*values)

    # Convert accuracies to percentage
    accuracies = [acc * 100 for acc in accuracies]
    standard_errors = [err * 100 for err in standard_errors]
    domain = np.linspace(1, len(words), len(words))

    # Split data into high and low accuracy points
    high_acc_indices = [i for i, acc in enumerate(accuracies) if acc >= 80]
    low_acc_indices = [i for i, acc in enumerate(accuracies) if acc < 80]

    high_acc_words = [words[i] for i in high_acc_indices]
    high_accuracies = [accuracies[i] for i in high_acc_indices]
    high_standard_errors = [standard_errors[i] for i in high_acc_indices]
    high_domain = [domain[i] for i in high_acc_indices]

    low_acc_words = [words[i] for i in low_acc_indices]
    low_accuracies = [accuracies[i] for i in low_acc_indices]
    low_standard_errors = [standard_errors[i] for i in low_acc_indices]
    low_domain = [domain[i] for i in low_acc_indices]

    # Specs for the figure
    plt.figure(figsize=(14, 9))
    width_ratios = [2.75, 1]
    gs = plt.GridSpec(1, 2, width_ratios=width_ratios)
    scatter_ax = plt.subplot(gs[0])

    # Plot scatter plot with error bars for high accuracy points
    scatter_ax.errorbar(high_domain, high_accuracies, yerr=high_standard_errors, fmt='o', capsize=5, color='#1f77b4', elinewidth=0.75, label='high', markersize=10)

    # Plot scatter plot with error bars for low accuracy points
    scatter_ax.scatter(low_domain, low_accuracies, color='red', label='low', marker='d', s=100)
    scatter_ax.errorbar(low_domain, low_accuracies, yerr=low_standard_errors, fmt='d', capsize=5, color='red', elinewidth=0.75, markersize=10)

    # Plotting the grey lines
    for i in range(1, len(words) + 1):
        plt.vlines(i, 0, 100, colors='gray', linestyles='dashed', alpha=0.5)

    # Format
    scatter_ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    legend_handle_1 = Line2D([], [], marker='o', markersize=10, label='high', color='#1f77b4')
    legend_handle_2 = Line2D([], [], marker='d', markersize=10, label='low', color='red')
    plt.legend(handles=[legend_handle_1, legend_handle_2], bbox_to_anchor=(1.01, 1.15), loc='upper right', fontsize=18, ncol=2, title='classifier accuracy:', title_fontproperties={'weight': 'bold', 'size': 18})
    scatter_ax.annotate("a)", xy=(0, 1.05), xycoords='axes fraction', fontsize=26, fontweight='bold')

    # Modify every other word
    all_words = list(words)
    for i in range(len(all_words)):
        if i % 2 == 1 and '-------' not in all_words[i]:
            all_words[i] += '-------'

    plt.xticks(domain, all_words, rotation=90, fontsize=15)
    scatter_ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    plt.yticks(fontsize=18)  # Adjust the fontsize as needed
    plt.xlabel('PBK Word Responses - Unaltered', fontweight='bold', fontsize=26)
    plt.ylabel('Percent Correct (%)', fontweight='bold', fontsize=26)
    plt.ylim(0, 100)

    # Create the bar plot
    kde_ax = plt.subplot(gs[1])
    y_kde = np.arange(0, 101, 10)

    kde_normals = gaussian_kde(accuracies)
    density_normals = kde_normals(y_kde)
    kde_ax.plot(density_normals, y_kde, color='#1f77b4', label='high', linewidth=2.5)
    
    # Add the second plot with the red color
    kde_normals_altered = gaussian_kde(low_accuracies)
    density_normals_altered = kde_normals_altered(y_kde)
    kde_ax.plot(density_normals_altered, y_kde, color='red', label='low', linestyle='dotted', linewidth=2.5)

    kde_ax.set_xlabel('Kernel Density', fontweight='bold', fontsize=26, labelpad=15)
    kde_ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    kde_ax.yaxis.set_tick_params(labelsize=18)
    kde_ax.xaxis.set_tick_params(labelsize=18)
    kde_ax.legend(loc='upper right', fontsize=18, bbox_to_anchor=(1.025, 1.15), ncol=2, title='classifier accuracy:', title_fontproperties={'weight': 'bold', 'size': 18})
    kde_ax.set_ylim(0, 100)  # Set y-axis limits from 0 to 100
    kde_ax.annotate("b)", xy=(0, 1.05), xycoords='axes fraction', fontsize=26, fontweight='bold')
    
    # Add x-axis to the KDE plot
    kde_ax.set_xlim(0, max(density_normals.max(), density_normals_altered.max()) * 1.1)  # Set x-axis limits
    kde_ax.set_xticks([tick for tick in kde_ax.get_xticks() if tick != 0.0])  # Filter out 0.0 tick
    tick_labels = [f'{tick:.2f}' for tick in kde_ax.get_xticks()]  # Format tick labels
    kde_ax.set_xticklabels(tick_labels)  # Set new tick labels

    plt.tight_layout()  
    plt.show()

    # Return words above 80% accuracy
    high_acc_words = [word.replace('-------', '') for word in high_acc_words]
    return high_acc_words

##############################
# Description: Creates accuracy plots with scatter and kernel density for PBK word responses.
# Parameters:
#   - data (dict): A dictionary mapping words to tuples of (accuracy, standard error).
#   - save_path (str): The file path where the plot image will be saved.
# Returns:
#   - list: Words with accuracy above 80% after plotting.
##############################
def create_multiple_acc_plot_unaltered(data_dict: dict, differentiate: bool) -> (dict, list):
    num_datasets = len(data_dict)
    
    # Create a figure with a single set of subplots
    fig = plt.figure(figsize=(14, 12))  # Adjust the height to make space for the tables
    # gs = GridSpec(4, 3, height_ratios=[1.5, 0.5, 0.3, 0.3], width_ratios=[4, 1.25, 0.8])
    gs = GridSpec(4, 3, height_ratios=[1.5, 0, 0, 0.2], width_ratios=[4, 1.25, 0.8])
    
    scatter_ax = plt.subplot(gs[0, 0])
    kde_ax = plt.subplot(gs[0, 1])
    spacer_ax = plt.subplot(gs[1, :])  # Spacer row
    # table_ax1 = plt.subplot(gs[2, :])  # Span the first table across all columns
    # table_ax2 = plt.subplot(gs[3, :])  # Span the second table across all columns

    # Hide the spacer axis
    spacer_ax.axis('off')

    colors = ['#203864', '#4472c4', '#00b0f0', '#ff0000']
    grey_border = '#ffffff'
    markers = ['o', 's', 'D', '^', 'v']
    
    legend_elements = []
    low_acc_words_dict = {}
    all_words = word_list[1] + word_list[2]  # Ensure word_list is defined

    # Normalization dictionary for word spellings
    normalization_dict = {'trey': 'tray'}

    # Calculate average accuracy for each word
    avg_accuracies = {}
    for word in all_words:
        accuracies = []
        for data in data_dict.values():
            if normalization_dict.get(word, word) in data:
                acc = data[normalization_dict.get(word, word)][0] * 100
                if acc is not None:
                    accuracies.append(acc)
        if accuracies:
            avg_accuracies[word] = sum(accuracies) / len(accuracies)
        else:
            avg_accuracies[word] = 0

    # Sort words based on average accuracy
    sorted_words = sorted(avg_accuracies.keys(), key=lambda x: avg_accuracies[x], reverse=True)

    word_index = {word: idx for idx, word in enumerate(sorted_words)}
    domain = np.arange(1, len(sorted_words) + 1)

    # # Create table data
    # table_data1_list = [["X-Axis Index"] + [str(i) for i in range(1, 51)], ["PBK Word"] + sorted_words[:50]]
    # table_data2_list = [["X-Axis Index"] + [str(i) for i in range(51, 101)], ["PBK Word"] + sorted_words[50:100]]
    
    for i, (label, data) in enumerate(data_dict.items()):
        # Remove '.json' from the label
        cleaned_label = label.replace('.json', '').replace(' ', '\n')
        
        filtered_data = {normalization_dict.get(word, word): data.get(normalization_dict.get(word, word), (None, None)) for word in sorted_words}

        accuracies = [filtered_data[word][0] * 100 if filtered_data[word][0] is not None else None for word in sorted_words]
        standard_errors = [filtered_data[word][1] * 100 if filtered_data[word][1] is not None else None for word in sorted_words]
        
        if differentiate:
            high_acc_indices = [i for i, acc in enumerate(accuracies) if acc is not None and acc >= 80]
            low_acc_indices = [i for i, acc in enumerate(accuracies) if acc is not None and acc < 80]
            
            high_acc_words = [sorted_words[i] for i in high_acc_indices]
            high_accuracies = [accuracies[i] for i in high_acc_indices]
            high_standard_errors = [standard_errors[i] for i in high_acc_indices]
            high_domain = [domain[i] for i in high_acc_indices]
            
            low_acc_words = [sorted_words[i] for i in low_acc_indices]
            low_accuracies = [accuracies[i] for i in low_acc_indices]
            low_standard_errors = [standard_errors[i] for i in low_acc_indices]
            low_domain = [domain[i] for i in low_acc_indices]

            color = colors[i % len(colors)]

            # Plot scatter plot with error bars for high accuracy points (circle markers)
            scatter_ax.scatter(high_domain, high_accuracies, color=color, edgecolor=grey_border, label=f'{cleaned_label} high', marker='o', s=140, linewidth=1.5)
            scatter_ax.errorbar(high_domain, high_accuracies, yerr=high_standard_errors, fmt='none', ecolor=color, elinewidth=0.75, capsize=5)

            # Plot scatter plot with error bars for low accuracy points (diamond markers)
            scatter_ax.scatter(low_domain, low_accuracies, color=color, edgecolor=grey_border, label=f'{cleaned_label} low', marker='d', s=140, linewidth=1.5)
            scatter_ax.errorbar(low_domain, low_accuracies, yerr=low_standard_errors, fmt='none', ecolor=color, elinewidth=0.75, capsize=5)

            if any(acc is not None for acc in accuracies):
                kde_normals = gaussian_kde([acc for acc in accuracies if acc is not None])
                density_normals = kde_normals(np.arange(0, 101, 10))
                kde_ax.plot(density_normals, np.arange(0, 101, 10), color=color, label=f'{cleaned_label} high', linewidth=2.5)

            if any (acc is not None for acc in low_accuracies):
                kde_normals_altered = gaussian_kde([acc for acc in low_accuracies if acc is not None])
                density_normals_altered = kde_normals_altered(np.arange(0, 101, 10))
                kde_ax.plot(density_normals_altered, np.arange(0, 101, 10), color=color, label=f'{cleaned_label} low', linestyle='dotted', linewidth=2.5)

            low_acc_words_dict[label] = [word.replace('-------', '') for word in low_acc_words]
        else:
            indices = [i for i, acc in enumerate(accuracies) if acc is not None]
            selected_words = [sorted_words[i] for i in indices]
            selected_accuracies = [accuracies[i] for i in indices]
            selected_standard_errors = [standard_errors[i] for i in indices]
            selected_domain = [domain[i] for i in indices]

            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]

            # Plot scatter plot with error bars for all accuracy points (same marker)
            scatter_ax.scatter(selected_domain, selected_accuracies, color=color, edgecolor=grey_border, label=f'{cleaned_label}', marker=marker, s=120, linewidth=1.5)
            scatter_ax.errorbar(selected_domain, selected_accuracies, yerr=selected_standard_errors, fmt='none', ecolor=color, elinewidth=0.75, capsize=5)

            if any(acc is not None for acc in selected_accuracies):                
                kde_normals = gaussian_kde([acc for acc in selected_accuracies if acc is not None])
                density_normals = kde_normals(np.arange(0, 101, 10))
                kde_ax.plot(density_normals, np.arange(0, 101, 10), color=color, label=f'{cleaned_label}', linewidth=2.5)

            # Create combined legend entry with marker and line
            legend_elements.append((Line2D([0], [0], marker=marker, color='w', label=f"{cleaned_label}", markerfacecolor=color, markersize=18),
                                    Line2D([0], [0], color=color, linewidth=4)))

    # Plotting the grey lines
    for i in range(1, len(sorted_words) + 1):
        scatter_ax.vlines(i, 0, 100, colors='gray', linestyles='dashed', alpha=0.5)

    # Format scatter plot
    scatter_ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    scatter_ax.annotate("A", xy=(0, 1.05), xycoords='axes fraction', fontsize=36, fontweight='bold')

    # Set x-axis ticks and labels
    scatter_ax.set_xticks(np.arange(0, 101, 1))
    scatter_ax.set_xticklabels(np.arange(0, 101, 1), fontsize=12)
    scatter_ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    scatter_ax.xaxis.set_tick_params(labelsize=18, rotation=90)
    scatter_ax.set_xlabel('PBK Word Index - Unaltered', fontweight='bold', fontsize=30)  # Increased font size
    scatter_ax.set_ylabel('Percent Correct', fontweight='bold', fontsize=30)
    scatter_ax.set_ylim(0, 100)
    scatter_ax.yaxis.set_tick_params(labelsize=18)

    # Format KDE plot
    kde_ax.set_xlabel('Kernel Density', fontweight='bold', fontsize=30, labelpad=15)  # Increased font size
    kde_ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    kde_ax.yaxis.set_tick_params(labelsize=18)
    kde_ax.xaxis.set_tick_params(labelsize=18)
    kde_ax.annotate("B", xy=(0, 1.05), xycoords='axes fraction', fontsize=36, fontweight='bold')

    kde_ax.set_ylim(0, 100)
    kde_ax.set_xlim(0, 0.04)  # Align x=0 with the y-axis
    kde_ax.set_xticks([tick for tick in kde_ax.get_xticks() if tick != 0.0])
    tick_labels = [f'{tick:.2f}' for tick in kde_ax.get_xticks()]
    kde_ax.set_xticklabels(tick_labels)

    # # Create table data using pandas and add an "X Value" row
    # table_data1 = pd.DataFrame({
    #     "X-Axis Index": [str(i) for i in range(1, 51)],
    #     "PBK Word": sorted_words[:50]
    # }).T

    # table_data2 = pd.DataFrame({
    #     "X-Axis Index": [str(i) for i in range(51, 101)],
    #     "PBK Word": sorted_words[50:100]
    # }).T

    # # Clear the axis for table_ax1 and table_ax2
    # table_ax1.axis('off')
    # table_ax2.axis('off')

    # # Function to rotate table and adjust height
    # def rotate_table(ax, df):
    #     # Add table to axis
    #     tab = table(ax, df, loc='center', cellLoc='center', colWidths=[0.03]*len(df.columns))
    #     tab.auto_set_font_size(False)
    #     tab.set_fontsize(10)
        
    #     # Adjust cell height and rotation
    #     for key, cell in tab.get_celld().items():
    #         cell.set_fontsize(10)  # Set font size for better readability
    #         if key[0] == 0:
    #             cell.set_height(0)
    #             cell.set_width(0.0215)
    #         if key[0] == 1:  # Increase height for 'X Value' row
    #             cell.set_height(0.4)
    #             cell.set_width(0.0215)
    #             if cell.get_text().get_text() == "X-Axis Index":
    #                 cell.get_text().set_fontweight('bold')
    #                 cell.get_text().set_horizontalalignment('center')
    #                 cell.get_text().set_fontsize(12)
    #             else:
    #                 cell.get_text().set_rotation(90)

    #         elif key[0] == 2:  # Increase height for 'PBK Word' row
    #             cell.set_height(0.8)
    #             cell.set_width(0.0215)
    #             if cell.get_text().get_text() == "PBK Word":
    #                 cell.get_text().set_fontweight('bold')
    #                 cell.get_text().set_horizontalalignment('center')
    #                 cell.get_text().set_fontsize(12)
    #             else:
    #                 cell.get_text().set_rotation(90)

    # # Remove column labels
    # table_data1.columns = [""] * len(table_data1.columns)
    # table_data2.columns = [""] * len(table_data2.columns)

    # # Rotate and add tables
    # rotate_table(table_ax1, table_data1)
    # rotate_table(table_ax2, table_data2)

    plt.subplots_adjust(hspace=0.1, top=0.9, bottom=0.05, left=0.1, right=0.95)

        # Manually adjust the position of the scatter plot
    scatter_ax.set_position([0.10, scatter_ax.get_position().y0, scatter_ax.get_position().width, scatter_ax.get_position().height])

    # # Manually adjust the position of the tables
    # table_ax1.set_position([0.15, table_ax1.get_position().y0, table_ax1.get_position().width-0.05, table_ax1.get_position().height])
    # table_ax2.set_position([0.15, table_ax2.get_position().y0, table_ax2.get_position().width-0.05, table_ax2.get_position().height])

    # Create a combined legend
    combined_legend_elements = []
    for marker_handle, line_handle in legend_elements:
        combined_legend_elements.append((marker_handle, line_handle))

    legend = fig.legend(handles=combined_legend_elements, labels=[h.get_label() for h, _ in legend_elements],
                        loc='center right', fontsize=20, framealpha=1.0, title='PBK scorer:', 
                        title_fontproperties={'weight': 'bold', 'size': 20}, handler_map={tuple: HandlerTuple(ndivide=None)})
    
    # Manually adjust the position of the legend
    # legend.set_bbox_to_anchor((1, 0.67))

    plt.show()

    return low_acc_words_dict, sorted_words

##############################
# Description: Generates scatter and kernel density plots for PBK word response accuracies.
# Parameters:
#   - data_1 (dict): Dictionary mapping words to tuples of (accuracy, standard error) for dataset 1.
#   - data_2 (dict): Dictionary mapping words to tuples of (accuracy, standard error) for dataset 2.
#   - color_1 (str): Color for dataset 1 plot elements.
#   - color_2 (str): Color for dataset 2 plot elements.
#   - words_above_80 (list): List of words with accuracy above 80%.
#   - label_1 (str): Label for dataset 1 in the legend.
#   - label_2 (str): Label for dataset 2 in the legend.
# Returns:
#   - None: The function displays plots but does not return values.
##############################
def create_acc_plot_altered(data_1: dict, data_2: dict, color_1: str, color_2: str, words_above_80: list, label_1: str, label_2: str) -> None:
    # Extract words, accuracies, and standard errors for both datasets
    accuracies_1 = [data_1[word][0] * 100 if word in data_1 else None for word in words_above_80]
    standard_errors_1 = [data_1[word][1] * 100 if word in data_1 else None for word in words_above_80]

    accuracies_2 = [data_2[word][0] * 100 if word in data_2 else None for word in words_above_80]
    standard_errors_2 = [data_2[word][1] * 100 if word in data_2 else None for word in words_above_80]

    # Define domain
    domain = np.linspace(1, len(words_above_80), len(words_above_80))

    # Specs for the figure
    plt.figure(figsize=(14, 9))
    width_ratios = [2.75, 1]
    gs = plt.GridSpec(1, 2, width_ratios=width_ratios)
    scatter_ax = plt.subplot(gs[0])

    # Plot scatter plot with error bars for data_1
    for i, (x, y, yerr) in enumerate(zip(domain, accuracies_1, standard_errors_1)):
        if y is not None:
            scatter_ax.errorbar(x, y, yerr=yerr, fmt='o', capsize=5, color=color_1, elinewidth=0.75, label=label_1 if i == 0 else "", markersize=10)

    # Plot scatter plot with error bars for data_2
    for i, (x, y, yerr) in enumerate(zip(domain, accuracies_2, standard_errors_2)):
        if y is not None:
            scatter_ax.errorbar(x, y, yerr=yerr, fmt='d', capsize=5, color=color_2, elinewidth=0.75, label=label_2 if i == 0 else "", markersize=10)

    # Plotting the grey lines
    for i in range(1, len(words_above_80) + 1):
        plt.vlines(i, 0, 100, colors='gray', linestyles='dashed', alpha=0.5)

    # Format
    scatter_ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    legend_handle_1 = Line2D([], [], marker='o', markersize=10, label=label_1, color=color_1)
    legend_handle_2 = Line2D([], [], marker='d', markersize=10, label=label_2, color=color_2)
    plt.legend(handles=[legend_handle_1, legend_handle_2], bbox_to_anchor=(1.01, 1.15), loc='upper right', fontsize=18, ncol=2, title='simulated condition:', title_fontproperties={'weight': 'bold', 'size': 18})
    scatter_ax.annotate("a)", xy=(0, 1.05), xycoords='axes fraction', fontsize=26, fontweight='bold')

    # Modify every other word
    for i in range(len(words_above_80)):
        if i % 2 == 1 and not words_above_80[i].endswith('-------'):
            words_above_80[i] += ' -------'

    plt.xticks(domain, words_above_80, rotation=90, fontsize=15)
    scatter_ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    plt.yticks(fontsize=18)  # Adjust the fontsize as needed
    plt.xlabel('PBK Word Responses - Altered', fontweight='bold', fontsize=26)
    plt.ylabel('Percent Correct', fontweight='bold', fontsize=26)
    plt.ylim(0, 100)

    # Create the bar plot
    kde_ax = plt.subplot(gs[1])
    y_kde = np.arange(0, 101, 10)

    kde_normals_1 = gaussian_kde([val for val in accuracies_1 if val is not None])
    density_normals_1 = kde_normals_1(y_kde)
    kde_ax.plot(density_normals_1, y_kde, color=color_1, label=label_1, linewidth=2.5)

    kde_normals_2 = gaussian_kde([val for val in accuracies_2 if val is not None])
    density_normals_2 = kde_normals_2(y_kde)
    kde_ax.plot(density_normals_2, y_kde, color=color_2, label=label_2, linestyle='dotted', linewidth=2.5)

    kde_ax.set_xlabel('Kernel Density', fontweight='bold', fontsize=26, labelpad=15)
    kde_ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    kde_ax.yaxis.set_tick_params(labelsize=18)
    kde_ax.xaxis.set_tick_params(labelsize=18)
    kde_ax.legend(loc='upper right', fontsize=18, bbox_to_anchor=(1.025, 1.15), ncol=2, title='simulated condition:', title_fontproperties={'weight': 'bold', 'size': 18})
    kde_ax.set_ylim(0, 100)  # Set y-axis limits from 0 to 100
    kde_ax.annotate("b)", xy=(0, 1.05), xycoords='axes fraction', fontsize=26, fontweight='bold')
    
    # Add x-axis to the KDE plot
    kde_ax.set_xlim(0, max(density_normals_1.max(), density_normals_2.max()) * 1.1)  # Set x-axis limits
    kde_ax.set_xticks([tick for tick in kde_ax.get_xticks() if tick != 0.0])  # Filter out 0.0 tick
    tick_labels = [f'{tick:.2f}' for tick in kde_ax.get_xticks()]  # Format tick labels
    kde_ax.set_xticklabels(tick_labels)  # Set new tick labels

    plt.tight_layout()  
    plt.show()

##############################
# Description: Generates multiple scatter and kernel density plots for PBK word response accuracies.
# Parameters:
#   - data_dict (dict): Dictionary mapping labels to data containing word accuracies and standard errors.
#   - low_acc_words_dict (dict): Dictionary mapping labels to lists of low accuracy words to exclude.
#   - all_words (list): List of all words to be plotted.
#   - condition (str): Condition label for the x-axis of the scatter plot.
# Returns:
#   - None: The function displays plots but does not return values.
##############################
def create_multiple_acc_plot_altered(data_dict: dict, low_acc_words_dict: dict, all_words: list, condition: str) -> None:
    fig = plt.figure(figsize=(14, 9))
    gs = plt.GridSpec(1, 2, width_ratios=[2.75, 1])
    
    scatter_ax = plt.subplot(gs[0])
    kde_ax = plt.subplot(gs[1])
    
    colors = ['#203864', '#4472c4', '#00b0f0', '#ff0000']
    markers = ['o', 's', 'D', '^']
    white_border = '#ffffff'
    
    legend_elements = []
    
    for i, (label, data) in enumerate(data_dict.items()):
        cleaned_label = label.replace('.json', '').replace(' ', '\n')
        
        words_to_plot = [word for word in all_words if word not in low_acc_words_dict.get(label, [])]
        
        accuracies = [data[word][0] * 100 if word in data else None for word in all_words]
        standard_errors = [data[word][1] * 100 if word in data else None for word in all_words]
        
        domain = np.linspace(1, len(all_words), len(all_words))

        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        scatter_ax.scatter([], [], color=color, edgecolor=white_border, marker=marker, s=140, linewidth=1.5)
        kde_ax.plot([], [], color=color, linewidth=2.5)
        
        for j, (x, y, yerr, word) in enumerate(zip(domain, accuracies, standard_errors, all_words)):
            if y is not None and word in words_to_plot:
                scatter_ax.scatter(x, y, color=color, edgecolor=white_border, marker=marker, s=140, linewidth=1.5)
                scatter_ax.errorbar(x, y, yerr=yerr, fmt='none', ecolor=color, elinewidth=0.75, capsize=5)
        
        valid_accuracies = [val for val in accuracies if val is not None and all_words[accuracies.index(val)] in words_to_plot]

        if len(set(valid_accuracies)) > 1:
            kde_normals = gaussian_kde(valid_accuracies)
            density_normals = kde_normals(np.arange(0, 101, 10))
            kde_ax.plot(density_normals, np.arange(0, 101, 10), color=color, linewidth=2.5)
        
        # Create combined legend entry with marker and line
        legend_elements.append((Line2D([0], [0], marker=marker, color='w', label=f"{cleaned_label}", markerfacecolor=color, markersize=18),
                                Line2D([0], [0], color=color, linewidth=4)))

    scatter_ax.vlines(range(1, len(all_words) + 1), 0, 100, colors='gray', linestyles='dashed', alpha=0.5)
    
    scatter_ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    scatter_ax.annotate("D", xy=(-0.1, 1.07), xycoords='axes fraction', fontsize=46, fontweight='bold')

    for k in range(len(all_words)):
        if k % 2 == 1 and not all_words[k].endswith(' -------'):
            all_words[k] += ' -------'

    # scatter_ax.set_xticks(domain)
    # scatter_ax.set_xticklabels(all_words, rotation=90, fontsize=12)
    scatter_ax.set_xticks(np.arange(0, 101, 1))
    scatter_ax.set_xticklabels(np.arange(0, 101, 1), fontsize=12)
    scatter_ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    scatter_ax.xaxis.set_tick_params(labelsize=30, rotation=90)

    scatter_ax.set_xlabel(f'PBK Word Index - {condition}', fontweight='bold', fontsize=40)
    scatter_ax.set_ylabel('Percent Correct', fontweight='bold', fontsize=40)
    scatter_ax.set_ylim(0, 100)
    scatter_ax.yaxis.set_tick_params(labelsize=30)

    kde_ax.set_xlabel('Kernel Density', fontweight='bold', fontsize=40, labelpad=15)
    kde_ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    kde_ax.yaxis.set_tick_params(labelsize=30)
    kde_ax.xaxis.set_tick_params(labelsize=30)
    kde_ax.set_ylim(0, 100)
    # kde_ax.annotate("b)", xy=(0, 1.05), xycoords='axes fraction', fontsize=24, fontweight='bold')
    
    kde_ax.set_xlim(0, 0.04)
    kde_ax.set_xticks([tick for tick in kde_ax.get_xticks() if tick != 0.0])
    tick_labels = [f'{tick:.2f}' for tick in kde_ax.get_xticks()]
    kde_ax.set_xticklabels(tick_labels)

    # Create a combined legend
    # combined_legend_elements = []
    # for marker_handle, line_handle in legend_elements:
    #     combined_legend_elements.append((marker_handle, line_handle))

    # fig.legend(handles=combined_legend_elements, labels=[h.get_label() for h, _ in legend_elements],
    #            loc='center right', fontsize=24, framealpha=1.0, title='PBK Evaluator:', title_fontproperties={'weight': 'bold', 'size': 24},
    #            handler_map={tuple: HandlerTuple(ndivide=None)})

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.show()

##############################
# Description: Creates a boxplot to visualize evaluation results for different classifiers and conditions.
# Parameters:
#   - data (dict): Dictionary with classifier-condition pairs as keys and corresponding data as values.
#   - classifiers (list): List of classifiers to be included in the boxplot.
#   - conditions (list): List of conditions for which results are plotted.
#   - colors (list): List of colors corresponding to each classifier.
#   - title (str): Title of the boxplot.
#   - ylabel (str): Label for the y-axis.
# Returns:
#   - None: The function displays the boxplot but does not return values.
##############################
def create_boxplot(data: dict, classifiers: list, conditions: list, colors: list, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    all_data = []
    labels = []
    color_patches = []

    for classifier, color in zip(classifiers, colors):
        for condition in conditions:
            condition_data = data.get((classifier, condition), [])
            all_data.append(condition_data)
            labels.append(f"{classifier} - {condition}")
            color_patches.append(color)

    box = ax.boxplot(all_data, patch_artist=True, widths=0.7)

    # Set colors
    for patch, color in zip(box['boxes'], color_patches):
        patch.set_facecolor(color)

    # Customize plot
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.set_ylabel(ylabel, fontweight='bold', fontsize=12)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=90, fontsize=10)

    # Add legend
    handles = [plt.Line2D([0], [0], color=color, lw=4) for color in colors]
    plt.legend(handles, classifiers, loc='upper right')

    plt.tight_layout()
    plt.show()

##############################
# Description: Evaluates and compiles performance metrics for various classifiers and conditions, saving results to a JSON file.
# Parameters: None
# Returns: None
# Outputs: Saves a JSON file containing evaluation results.
##############################
def get_evaluation_results():
    ### Alternative Evaluation ###
    master_lists = ['speechmatics.json', 'whisper large.json', 'whisper medium.json']
    people = ['aCauchi', 'Claire', 'Cole', 'Daniel', 'Hillary', 'Lulia', 'Melissa', 'Micheal', 'Polonenko', 'rCauchi', 'Robel', 'Samsoor']
    conditions = ['dFirst', 'dLast', 'c1000', 'c2000', 'c4000', 'c6000']

    results = {}
    for master in master_lists:
        results[master] = {}
        L1_normal, L2_normal = extract_data(master, condition='normal')
        normal_scores = score_condition_by_person(L1_normal, L2_normal, people)

        true_values = determine_P_N(positive=True, data=normal_scores)
        TP = true_values['TP']
        FN = true_values['FN']

        for condition in conditions:
            L1_c1, L2_c1 = extract_data(master, condition=condition)
            c1_scores = score_condition_by_person(L1_c1, L2_c1, people)

            if condition == 'dFirst':
                words_removed = ['are', 'is', 'end', 'own', 'axe', 'all', 'as', 'on']
                c1_scores = modify_scores(c1_scores, words_removed)

            elif condition == 'dLast':
                words_removed = ['no', 'tree', 'lay', 'me', 'few', 'plow', 'gray', 'bee', 'grew', 'knee', 'tray']
                c1_scores = modify_scores(c1_scores, words_removed)
            
            c1_values = determine_P_N(positive=False, data=c1_scores)

            TN = c1_values['TN']
            FP = c1_values['FP']

            accuracy = determine_accuracy(TP=TP, TN=TN, FP=FP, FN=FN)
            precision = determine_precision(TP=TP, FP=FP)
            recall = determine_recall(TP=TP, FN=FN)
            fscore = determine_fscore(precision=precision, recall=recall)

            results[master][condition] = {
                'TP': TP,
                'FN': FN,
                'TN': TN,
                'FP': FP, 
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'fscore': fscore
            }

    inter_rater = {}
    new_data = re_order_interrater()
    normal = new_data['Normal']
    dFirst = new_data['dFirst']
    dLast = new_data['dLast']
    c1000 = new_data['c1000']
    c2000 = new_data['c2000']
    c4000 = new_data['c4000']
    c6000 = new_data['c6000']

    negative_condition_dicts = [normal, dFirst, dLast, c1000, c2000, c4000, c6000]
    condition_names = ['dFirst', 'dLast', 'c1000', 'c2000', 'c4000', 'c6000']

    true_values = determine_P_N(positive=True, data=normal)
    TP = true_values['TP']
    FN = true_values['FN']

    for condition_name, condition_dict in zip(condition_names, negative_condition_dicts):
        
        inter_rater[condition_name] = {}
        false_values = determine_P_N(positive=False, data=condition_dict)
        TN = false_values['TN']
        FP = false_values['FP']

        accuracy = determine_accuracy(TP=TP, TN=TN, FP=FP, FN=FN)
        precision = determine_precision(TP=TP, FP=FP)
        recall = determine_recall(TP=TP, FN=FN)
        fscore = determine_fscore(precision=precision, recall=recall)

        inter_rater[condition_name] = {
                'TP': TP,
                'FN': FN,
                'TN': TN,
                'FP': FP, 
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'fscore': fscore
            }
        
    print(inter_rater)
        
    results['Inter Rater'] = inter_rater

    # Save results to a JSON file
    with open('data/evaluation_results.json', 'w') as json_file:
        json.dump(results, json_file, indent=4)

    print('Evaluation results saved to evaluation_results.json')

##############################
# Description: Plots specified metrics from evaluation results for different models and conditions, saving the plot as a PNG file.
# Parameters: 
#   metric (str): The metric to plot (e.g., 'accuracy', 'precision').
#   file_prefix (str): The prefix for the output plot filename (default is 'plot').
# Returns: None
# Outputs: Saves a PNG file containing the bar plot of the metrics.
##############################
def plot_metrics(metric, file_prefix='plot') -> None:
    file_path = 'data/evaluation_results.json'
    with open(file_path, 'r') as file:
        data = json.load(file)

    conditions = list(next(iter(data.values())).keys())
    models = list(data.keys())
    
    metrics = {condition: [] for condition in conditions}
    
    for model in models:
        for condition in conditions:
            metrics[condition].append(data[model][condition][metric])
    
    bar_width = 0.15
    index = np.arange(len(conditions))
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for i, model in enumerate(models):
        ax.bar(index + i * bar_width, [metrics[condition][i] for condition in conditions], bar_width, label=model)
    
    ax.set_xlabel('Conditions')
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f'{metric.capitalize()} for Different Conditions and Models')
    ax.set_xticks(index + bar_width * (len(models) - 1) / 2)
    ax.set_xticklabels(conditions)
    ax.legend()
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{file_prefix}_{metric}.png')
    plt.close()
    plt.show()

##############################
# Description: Evaluates human accuracy for words under different conditions and calculates accuracy and standard deviation for each word.
# Returns: 
#   dict: A dictionary containing the accuracy and standard deviation for each word under each condition.
##############################
def eval_human_for_big_plots() -> dict:

    humans = ['cole', 'joyce', 'ryley', 'lulia', 'isabel', 'susan', 'maria']
    # Load the data
    with open('data/inter_rater_assess.json', 'r') as file:
        data = json.load(file)

    # Initialize the new structure
    condition_word_acc_dict = {}
    conditions = ['Normal', 'dFirst', 'dLast', 'c1000', 'c2000', 'c4000', 'c6000']

    for condition in conditions:
        word_acc_dict = defaultdict(list)
        data_subset = [entry for entry in data if entry.get('condition') == condition]
        
        # Identify and print duplicate words
        word_counts = Counter(entry['word'] for entry in data_subset)
        duplicates = {word: count for word, count in word_counts.items() if count > 1}
        # if duplicates:
        #     print("Duplicate words and their counts:")
        #     for word, count in duplicates.items():
        #         print(f"{word}: {count}")

        # Accumulate scores for each word
        for entry in data_subset:
            if 'word number' in entry:
                del entry['word number']
            if 'speaker' in entry:
                del entry['speaker']

            target_word = entry['word']
            responses = entry['responses']

            scores_by_word = []
            for human in humans:
                if responses[human] == target_word:
                    scores_by_word.append(1)
                else:
                    scores_by_word.append(0)

            word_acc_dict[target_word].extend(scores_by_word)

        # Calculate combined accuracy and standard deviation for each word
        final_word_acc_dict = {}
        for word, scores in word_acc_dict.items():
            acc = round(np.mean(scores), 2)
            std = round(np.std(scores), 2)
            final_word_acc_dict[word] = (acc, std)

        condition_word_acc_dict[condition] = final_word_acc_dict

    return condition_word_acc_dict

##############################
# Description: Generates a point plot of overall accuracy with error bars for different conditions.
# Returns: 
#   None
##############################
def overall_accuracy_plot() -> None:

    file_path = '/Users/ayden/Desktop/SickKids/excel/data_long/emm_all_df.xlsx'

# Load the data
    data = pd.read_excel(file_path)
    
    # Set the plot style
    sns.set(style="whitegrid")
    
    # Create the plot
    plt.figure(figsize=(16, 10))
    
    # Generate the point plot without lines
    plot = sns.pointplot(data=data, x='condition', y='emmean', hue='listener', 
                         markers=['o', 's', 'D', '^'], linestyles='', dodge=0.3, ci=None, 
                         palette='deep', scale=3)
    
    # Add error bars (1 SE above and below each point)
    for idx, row in data.iterrows():
        plt.errorbar(x=idx // 4 + 0.1*(idx % 4 - 1.5), y=row['emmean'], yerr=row['SE']*2, fmt='none', 
                     capsize=5, color=plot.get_lines()[idx].get_color(), lw=2)
    
    # Set x and y axis labels with increased font size and bold text
    plot.set_xlabel('PBK Alteration Condition', fontsize=20, fontweight='bold')
    plot.set_ylabel('Percent Correct (%)', fontsize=20, fontweight='bold')
    
    # Set x-ticks labels with increased font size
    plot.set_xticklabels(['Unaltered', 'First Consonant \nDeletion', 'Last Consonant \nDeletion', 
                          'LPF 1 kHz', 'LPF 2 kHz', 'LPF 4 kHz', 'LPF 6 kHz'], rotation=45, fontsize=16)
    
    # Set y-ticks with increased font size and every 10%
    plt.yticks(range(0, 101, 10), fontsize=16)
    
    # Increase the legend font size
    plt.legend(title='Listener', title_fontsize='13', fontsize='12')
    
    # Display the plot
    plt.tight_layout()
    plt.show()

##############################
# Description: Calculates and prints Fleiss' kappa for human and machine ratings.
# Parameters:
#   classifier (str): The name of the classifier to be evaluated.
# Returns: 
#   None
##############################
def get_fleis_kappa(classifier: str) -> None:

    human_raters = ['cole', 'joyce', 'ryley', 'lulia', 'isabel', 'susan', 'maria']

    people_ratings = {
        'cole': [],
        'joyce': [],
        'ryley': [],
        'lulia': [],
        'isabel': [],
        'susan': [],
        'maria': []
    }
    machine_ratings = []

    with open('data/inter_rater_assess.json', 'r') as inter_rater_json:
        inter_rater_data = json.load(inter_rater_json)

    with open(f'data/json/master/{classifier}', 'r') as classifier_json:
        classifier_data = json.load(classifier_json)

    for entry in inter_rater_data:
        speaker = entry['speaker']
        condition = entry['condition']
        if condition == 'Normal':
            condition = 'normal'
        target_word = entry["word"]
        word_number = entry['word number']

        if word_number < 50:
            list_number = 'L1'
            concat_word_number = word_number
        else:
            list_number = 'L2'
            concat_word_number = word_number - 50

        machine_response = classifier_data[f"{list_number}_{condition}"][speaker][concat_word_number]
        if machine_response == target_word:
            machine_bool = 1
        else:
            machine_bool = 0

        machine_ratings.append(machine_bool)

        for human_rater in human_raters:
            rated_word = entry['responses'][human_rater]
            if rated_word == target_word:
                human_bool = 1
            else: 
                human_bool = 0

            people_ratings[human_rater].append(human_bool)

    all_ratings = []
    all_ratings.append(machine_ratings)

    for human in human_raters:
        all_ratings.append(people_ratings[human])

    # Transpose the list to get ratings per item
    all_ratings = np.array(all_ratings).T.tolist()

    # Convert to required format for Fleiss' kappa
    n_items = len(all_ratings)
    counts = np.zeros((n_items, 2))
    for i, ratings in enumerate(all_ratings):
        counts[i, 0] = ratings.count(0)
        counts[i, 1] = ratings.count(1)

    # Calculate Fleiss' kappa using statsmodels
    kappa = ir.fleiss_kappa(counts, method='fleiss')

    print("Fleiss' kappa:", kappa)
    
##############################
# Description: Calculates Fleiss' kappa for combined ratings of specified classifiers and human raters.
# Parameters:
#   classifiers (list): List of classifier names to evaluate.
#   human_raters (list): List of human raters' names.
#   raters_to_evaluate (list): List of raters for which Fleiss' kappa is calculated.
# Returns: 
#   float: Fleiss' kappa value.
##############################
def get_combined_fleiss_kappa(classifiers: list, human_raters: list, raters_to_evaluate: list) -> float:
    # Initialize a dictionary to hold ratings for each classifier and human rater
    ratings_dict = {classifier: [] for classifier in classifiers}
    ratings_dict.update({rater: [] for rater in human_raters})

    with open('data/inter_rater_assess.json', 'r') as inter_rater_json:
        inter_rater_data = json.load(inter_rater_json)

    for classifier in classifiers:
        with open(f'data/json/master/{classifier}', 'r') as classifier_json:
            classifier_data = json.load(classifier_json)

        for entry in inter_rater_data:
            speaker = entry['speaker']
            condition = entry['condition']
            if condition == 'Normal':
                condition = 'normal'
            target_word = entry["word"]
            word_number = entry['word number']

            if word_number < 50:
                list_number = 'L1'
                concat_word_number = word_number
            else:
                list_number = 'L2'
                concat_word_number = word_number - 50

            machine_response = classifier_data[f"{list_number}_{condition}"][speaker][concat_word_number]
            machine_bool = 1 if machine_response == target_word else 0
            ratings_dict[classifier].append(machine_bool)

    for entry in inter_rater_data:
        target_word = entry["word"]

        for human_rater in human_raters:
            rated_word = entry['responses'][human_rater]
            human_bool = 1 if rated_word == target_word else 0
            ratings_dict[human_rater].append(human_bool)

    # Prepare all_ratings for the specified raters
    all_ratings = []
    for rater in raters_to_evaluate:
        all_ratings.append(ratings_dict[rater])

    # Transpose the list to get ratings per item
    all_ratings = np.array(all_ratings).T.tolist()

    # Convert to required format for Fleiss' kappa
    n_items = len(all_ratings)
    counts = np.zeros((n_items, 2))
    for i, ratings in enumerate(all_ratings):
        counts[i, 0] = ratings.count(0)
        counts[i, 1] = ratings.count(1)

    # print(counts)
    # Calculate Fleiss' kappa using statsmodels
    kappa = ir.fleiss_kappa(counts, method='fleiss')

    return kappa

##############################
# Description: Calculates the percent agreement among specified classifiers and human raters.
# Parameters:
#   classifiers (list): List of classifier names to evaluate.
#   human_raters (list): List of human raters' names.
#   raters_to_evaluate (list): List of raters for which percent agreement is calculated.
# Returns: 
#   None: Prints the percent agreement value.
##############################
def get_percent_agreement(classifiers: list, human_raters: list, raters_to_evaluate: list) -> None:
    # Initialize a dictionary to hold ratings for each classifier and human rater
    ratings_dict = {classifier: [] for classifier in classifiers}
    ratings_dict.update({rater: [] for rater in human_raters})

    with open('data/inter_rater_assess.json', 'r') as inter_rater_json:
        inter_rater_data = json.load(inter_rater_json)

    for classifier in classifiers:
        with open(f'data/json/master/{classifier}', 'r') as classifier_json:
            classifier_data = json.load(classifier_json)

        for entry in inter_rater_data:
            speaker = entry['speaker']
            condition = entry['condition']
            if condition == 'Normal':
                condition = 'normal'
            target_word = entry["word"]
            word_number = entry['word number']

            if word_number < 50:
                list_number = 'L1'
                concat_word_number = word_number
            else:
                list_number = 'L2'
                concat_word_number = word_number - 50

            machine_response = classifier_data[f"{list_number}_{condition}"][speaker][concat_word_number]
            machine_bool = 1 if machine_response == target_word else 0
            ratings_dict[classifier].append(machine_bool)

    for entry in inter_rater_data:
        target_word = entry["word"]

        for human_rater in human_raters:
            rated_word = entry['responses'][human_rater]
            human_bool = 1 if rated_word == target_word else 0
            ratings_dict[human_rater].append(human_bool)

    # Prepare all_ratings for the specified raters
    all_ratings = []
    for rater in raters_to_evaluate:
        all_ratings.append(ratings_dict[rater])

    # Transpose the list to get ratings per item
    all_ratings = np.array(all_ratings).T.tolist()

    # Calculate percent agreement
    n_items = len(all_ratings)
    n_agreements = 0

    for ratings in all_ratings:
        if len(set(ratings)) == 1:
            n_agreements += 1

    percent_agreement = (n_agreements / n_items) * 100

    print(f"Percent Agreement: {percent_agreement:.2f}%")

##############################
# Description: Calculates the percent agreement among specified classifiers and human raters for given conditions.
# Parameters:
#   classifiers (list): List of classifier names to evaluate.
#   human_raters (list): List of human raters' names.
#   raters_to_evaluate (list): List of raters for which percent agreement is calculated.
#   conditions_to_consider (list): List of conditions to consider for agreement calculation.
# Returns: 
#   float: The percent agreement value rounded to two decimal places.
##############################
def get_percent_agreement_specific(classifiers: list, human_raters: list, raters_to_evaluate: list, conditions_to_consider:list) -> float:
    # Initialize a dictionary to hold ratings for each classifier and human rater
    ratings_dict = {classifier: [] for classifier in classifiers}
    ratings_dict.update({rater: [] for rater in human_raters})

    with open('data/inter_rater_assess.json', 'r') as inter_rater_json:
        inter_rater_data = json.load(inter_rater_json)

    for classifier in classifiers:
        with open(f'data/json/master/{classifier}', 'r') as classifier_json:
            classifier_data = json.load(classifier_json)

        for entry in inter_rater_data:
            speaker = entry['speaker']
            condition = entry['condition']
            if condition == 'Normal':
                condition = 'normal'
            if condition not in conditions_to_consider:
                continue

            target_word = entry["word"]
            word_number = entry['word number']

            if word_number < 50:
                list_number = 'L1'
                concat_word_number = word_number
            else:
                list_number = 'L2'
                concat_word_number = word_number - 50

            machine_response = classifier_data[f"{list_number}_{condition}"][speaker][concat_word_number]
            machine_bool = 1 if machine_response == target_word else 0
            ratings_dict[classifier].append(machine_bool)

    for entry in inter_rater_data:
        condition = entry['condition']
        if condition == 'Normal':
            condition = 'normal'
        if condition not in conditions_to_consider:
            continue

        target_word = entry["word"]

        for human_rater in human_raters:
            rated_word = entry['responses'][human_rater]
            human_bool = 1 if rated_word == target_word else 0
            ratings_dict[human_rater].append(human_bool)

    # Prepare all_ratings for the specified raters
    all_ratings = []
    for rater in raters_to_evaluate:
        all_ratings.append(ratings_dict[rater])

    # Transpose the list to get ratings per item
    all_ratings = np.array(all_ratings).T.tolist()

    # Calculate percent agreement
    n_items = len(all_ratings)
    n_agreements = 0

    for ratings in all_ratings:
        if len(set(ratings)) == 1:
            n_agreements += 1

    percent_agreement = (n_agreements / n_items) * 100

    return round(percent_agreement, 2)

##############################
# Description: Computes the combined Fleiss' kappa for specified classifiers and human raters under given conditions.
# Parameters:
#   classifiers (list): List of classifier names to evaluate.
#   human_raters (list): List of human raters' names.
#   raters_to_evaluate (list): List of raters for which Fleiss' kappa is calculated.
#   conditions_to_consider (list): List of conditions to consider for kappa calculation.
# Returns: 
#   float: The combined Fleiss' kappa value rounded to two decimal places.
##############################
def get_combined_fleiss_kappa_specific(classifiers: list, human_raters: list, raters_to_evaluate: list, conditions_to_consider:list) -> float:
    # Initialize a dictionary to hold ratings for each classifier and human rater
    ratings_dict = {classifier: [] for classifier in classifiers}
    ratings_dict.update({rater: [] for rater in human_raters})

    with open('data/inter_rater_assess.json', 'r') as inter_rater_json:
        inter_rater_data = json.load(inter_rater_json)

    for classifier in classifiers:
        with open(f'data/json/master/{classifier}', 'r') as classifier_json:
            classifier_data = json.load(classifier_json)

        for entry in inter_rater_data:
            speaker = entry['speaker']
            condition = entry['condition']
            if condition == 'Normal':
                condition = 'normal'
            if condition not in conditions_to_consider:
                continue

            target_word = entry["word"]
            word_number = entry['word number']

            if word_number < 50:
                list_number = 'L1'
                concat_word_number = word_number
            else:
                list_number = 'L2'
                concat_word_number = word_number - 50

            machine_response = classifier_data[f"{list_number}_{condition}"][speaker][concat_word_number]
            # machine_bool = 1 if machine_response == target_word else 0
            if machine_response == target_word:
                machine_bool = 1
            else:
                machine_bool = 0
                # print(f'Target Word: {target_word}; Machine Res: {machine_response}')

            ratings_dict[classifier].append(machine_bool)

    for entry in inter_rater_data:
        condition = entry['condition']
        if condition == 'Normal':
            condition = 'normal'
        if condition not in conditions_to_consider:
            continue

        target_word = entry["word"]

        for human_rater in human_raters:
            rated_word = entry['responses'][human_rater]
            human_bool = 1 if rated_word == target_word else 0
            ratings_dict[human_rater].append(human_bool)

    # Prepare all_ratings for the specified raters
    all_ratings = []
    for rater in raters_to_evaluate:
        all_ratings.append(ratings_dict[rater])

    # Transpose the list to get ratings per item
    all_ratings = np.array(all_ratings).T.tolist()
    # print(all_ratings)
    # Convert to required format for Fleiss' kappa
    n_items = len(all_ratings)
    counts = np.zeros((n_items, 2))
    for i, ratings in enumerate(all_ratings):
        counts[i, 0] = ratings.count(0)
        counts[i, 1] = ratings.count(1)

    # Calculate Fleiss' kappa using statsmodels
    kappa = ir.fleiss_kappa(counts, method='fleiss')

    return round(kappa,2)
    # print(f"Fleiss' kappa: {kappa:.6f}")
