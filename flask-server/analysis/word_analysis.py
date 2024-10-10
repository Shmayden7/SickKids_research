#########################
# Imports:
import json
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import pandas as pd
import statsmodels.api as sm

# from rpy2 import robjects as ro
# import rpy2.robjects as robjects
# from rpy2.robjects import r
# from rpy2.robjects.packages import importr

from data.constants import word_list
from analysis.json_helpers import loadDictFromJSON, pullDPrimeMetricsByPerson
from classes.DoubleyLinkedList import Node, DoubleyLinkedList
from analysis.analysis_functions import loadXFromSavedJson

# Constants:
#########################

##############################
# Description: Segments responses from a JSON file based on specified parameters.
# Parameters: 
#   - eval_file: The name of the evaluation JSON file (str).
#   - confidence: A threshold value for filtering responses (int).
#   - audio_path: Path to the audio files directory (str, optional).
#   - json_path: Path to the JSON files directory (str, optional).
#   - segment_dump_location: Path to store segmented audio files (str, optional).
# Returns: None
##############################
def segmentResponces(eval_file: str, confadince: int, audio_path='data/audio/', json_path='data/json/', segment_dump_location='data/audio/segment/') -> None:
    import_json = loadDictFromJSON(eval_file, json_path)
    import_json_words = import_json['words']
    ideal_word_list = word_list[import_json['list_number']]
    
    # Initiating the Doubly Linked List
    robot_prompts = DoubleyLinkedList()
    child_responces = DoubleyLinkedList()

    # Populating the Lists
    for word in import_json_words:
        if word['speaker'] == 'A':
            robot_prompts.push_tail(word)
        elif word['speaker'] == 'B':
            child_responces.push_tail(word)

##############################
# Description: Computes the average accuracy and standard error for words from JSON data.
# Parameters: 
#   - list_number: The index of the word list to analyze (int).
#   - json_path: The path to the JSON files directory (str).
# Returns: 
#   - accuracy: List of average accuracies per word (list of floats).
#   - error: List of standard errors for each word's accuracy (list of floats).
##############################
def getAveragedAccByWord(list_number: int, json_path: str):
    accData = loadXFromSavedJson('accuracy', list_number, json_path)
    words = word_list[list_number]

    # organizing the data via word not via trial
    twoDim_acc_data = [
        [],[],[],[],[],[],[],[],[],[],
        [],[],[],[],[],[],[],[],[],[],
        [],[],[],[],[],[],[],[],[],[],
        [],[],[],[],[],[],[],[],[],[],
        [],[],[],[],[],[],[],[],[],[],
    ]
    for x in range(len(words)):
        for normal in accData:
            sub = normal['data']
            twoDim_acc_data[x].append(sub[x])

    accuracy = []
    for sub in twoDim_acc_data:
        accuracy.append(sum(sub) / len(sub))

    error = []
    for sub in twoDim_acc_data:
        s_div = np.std(sub, ddof=1)
        standard_error = s_div / np.sqrt(len(sub))
        error.append(standard_error)

    return accuracy, error


def reOrderWordsByAccuracy(data: dict, words: dict, err: dict):

    # if the array is 2D - i.e. if there are multiple conditions
    if isinstance(data, list) and all(isinstance(row, list) for row in data):
        mean = []
        for subarray in data:
            for i in range(len(subarray)):
                if len(mean) < len(subarray):
                    mean.append(subarray[i])
                else:
                    mean[i] = (mean[i]+subarray[i])/2
        
        enumerated_mean = list(enumerate(mean))
        sorted_mean = sorted(enumerated_mean, key=lambda x: x[1], reverse=True)

        sorted_indices = [index for index, _ in sorted_mean]

        sorted_words = [words[i] for i in sorted_indices]
        sorted_numbers = []
        for subarray in data:
            sorted_sub = [subarray[i] for i in sorted_indices]
            sorted_numbers.append(sorted_sub)        
    else:
        combined = list(zip(data, words, err))
        combined.sort(key=lambda x: x[0], reverse=True)
        sorted_numbers, sorted_words, err = zip(*combined)
    
    return sorted_numbers, sorted_words, err

def reOrderByWords(accuracy, words, error, slice_val, new_order):
    # Check if the new_order array has the same length as words
    if len(new_order) != len(words):
        raise ValueError("Length of 'new_order' array should be equal to the length of 'words' array.")
    
    # Convert accuracy, error, and words to numpy arrays
    accuracy = np.array(accuracy)
    error = np.array(error)
    words = np.array(words)

    # Create a mapping dictionary for unique words to their respective indices
    word_to_index = {word: index for index, word in enumerate(np.unique(words))}
    
    # Use the mapping dictionary to get the indices corresponding to the order specified in new_order
    sorted_indices = np.array([word_to_index[word] for word in new_order])
    
    # Sort the accuracy, error, and words arrays based on the sorted_indices
    sorted_accuracy = accuracy[sorted_indices]
    sorted_error = error[sorted_indices]
    sorted_words = new_order

    # Handle the slice_val to remove elements from the end of the arrays
    if slice_val < 0:
        raise ValueError("Slice value should be non-negative.")
    parsed_accuracy = sorted_accuracy[:-slice_val]
    parsed_error = sorted_error[:-slice_val]
    parsed_words = sorted_words[:-slice_val]

    return parsed_accuracy, parsed_words, parsed_error

def plotMeanAccByWord():
    words = word_list[1] + word_list[2]

    L1_normals, L1_normals_err = getAveragedAccByWord(1, 'data/json/segmented_norms/L1_normal/')
    L2_normals, L2_normals_err = getAveragedAccByWord(2, 'data/json/segmented_norms/L2_normal/')
    normals = L1_normals+L2_normals
    normals = [(round(value,2)*100) for value in normals]

    normals_err = L1_normals_err + L2_normals_err

    normals_err = [(round(value,2)*100) for value in normals_err]
    normals, normals_words, normals_err = reOrderWordsByAccuracy(normals, words, normals_err)

    normals_order = ['great', 'pinch', 'such', 'bus', 'need', 'ways', 'mouth', 'fed', 'hunt', 'are', 'teach', 'slice', 'is', 'smile', 'scab', 'me', 'beef', 'shop', 'page', "weed", 'park', 'wait', 'fat', 'cage', 'turn', 'grab', 'rose', 'be', 'his', 'suit', 'splash', 'path', 'feed', 'next', 'wreck', 'waste', 'peg', 'freeze', 'race', 'fair', 'as', 'grew', 'cat', 'find', 'yes', 'please', 'sled', 'bad', 'five', 'fold', 'no', 'bath', 'slip', 'pink', 'thank', 'neck', 'few', 'use', 'did', 'pond', 'hot', 'laugh', 'falls', 'paste', 'plow', 'gray', 'lip', 'fresh', 'trey', 'on', 'camp', 'loud', 'put', 'box', 'take', 'lay', 'class', 'dish', 'bead', 'axe', 'knife', 'sing', 'all', 'bless', 'crab', 'darn', 'sack', 'pants', 'rat', 'tree', 'cart', 'bit', 'got', 'own', 'bud', 'knee', 'rag', 'ride', 'bet', 'end']

    # L1_dFirst, L1_dFirst_err = getAveragedAccByWord(1, 'data/json/segmented_norms/L1_dFirst/')    
    # L2_dFirst, L2_dFirst_err = getAveragedAccByWord(2, 'data/json/segmented_norms/L2_dFirst/')
    # dFirst = L1_dFirst+L2_dFirst
    # dFirst = [(round(value,2)*100) for value in dFirst]
    # dFirst_err = L1_dFirst_err + L2_dFirst_err
    # dFirst_err = [(round(value,2)*100) for value in dFirst_err]
    # dFirst, dFirst_words, dFirst_err = reOrderByWords(dFirst, words, dFirst_err, 13, normals_order)

    # L1_dLast, L1_dLast_err = getAveragedAccByWord(1, 'data/json/segmented_norms/L1_dLast/')    
    # L2_dLast, L2_dLast_err = getAveragedAccByWord(2, 'data/json/segmented_norms/L2_dLast/')
    # dLast = L1_dLast+L2_dLast
    # dLast = [(round(value,2)*100) for value in dLast]
    # dLast_err = L1_dLast_err+L2_dLast_err
    # dLast_err = [(round(value,2)*100) for value in dLast_err]
    # dLast, dLast_words, dLast_err = reOrderByWords(dLast, words, dLast_err, 13, normals_order)


    # L1_c1000, L1_c1000_err = getAveragedAccByWord(1, 'data/json/segmented_norms/L1_c1000/')
    # L2_c1000, L2_c1000_err = getAveragedAccByWord(2, 'data/json/segmented_norms/L2_c1000/')
    # c1000 = L1_c1000+L2_c1000
    # c1000 = [(round(value,2)*100) for value in c1000]
    # c1000_err = L1_c1000_err+L2_c1000_err
    # c1000_err = [(round(value,2)*100) for value in c1000_err]
    # c1000, c1000_words, c1000_err = reOrderByWords(c1000, words, c1000_err, 13, normals_order)

    # L1_c2000, L1_c2000_err = getAveragedAccByWord(1, 'data/json/segmented_norms/L1_c2000/')
    # L2_c2000, L2_c2000_err = getAveragedAccByWord(2, 'data/json/segmented_norms/L2_c2000/')
    # c2000 = L1_c2000+L2_c2000
    # c2000 = [(round(value,2)*100) for value in c2000]
    # c2000_err = L1_c2000_err+L2_c2000_err
    # c2000_err = [(round(value,2)*100) for value in c2000_err]
    # c2000, c2000_words, c2000_err = reOrderByWords(c2000, words, c2000_err, 13, normals_order)

    # L1_c4000, L1_c4000_err = getAveragedAccByWord(1, 'data/json/segmented_norms/L1_c4000/')
    # L2_c4000, L2_c4000_err = getAveragedAccByWord(2, 'data/json/segmented_norms/L2_c4000/')
    # c4000 = L1_c4000+L2_c4000
    # c4000 = [(round(value,2)*100) for value in c4000]
    # c4000_err = L1_c4000_err+L2_c4000_err
    # c4000_err = [(round(value,2)*100) for value in c4000_err]
    # c4000, c4000_words, c4000_err = reOrderByWords(c4000, words, c4000_err, 13, normals_order)

    # L1_c6000, L1_c6000_err = getAveragedAccByWord(1, 'data/json/segmented_norms/L1_c6000/')
    # L2_c6000, L2_c6000_err = getAveragedAccByWord(2, 'data/json/segmented_norms/L2_c6000/')
    # c6000 = L1_c6000+L2_c6000
    # c6000 = [(round(value,2)*100) for value in c6000]
    # c6000_err = L1_c6000_err+L2_c6000_err
    # c6000_err = [(round(value,2)*100) for value in c6000_err]
    # c6000, c6000_words, c6000_err = reOrderByWords(c6000, words, c6000_err, 13, normals_order)

    # sorted_numbers, words = reOrderWordsByAccuracy([c1000,c2000,c4000,c6000], words)
    # c1000 = sorted_numbers[0]
    # c1000 = sorted_numbers[0]
    # c1000 = sorted_numbers[0]
    # c1000 = sorted_numbers[0]

    plt.figure(figsize=(14, 9))
    width_ratios = [2.75, 1]
    gs = plt.GridSpec(1, 2, width_ratios=width_ratios)

    scatter_ax = plt.subplot(gs[0])

    #colours = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    colours = ['#1f77b4']
    #scatter_data = [normals,dFirst,dLast,c1000,c2000,c4000,c6000]
    scatter_data = [normals]
    #error_data = [normals_err,dFirst_err,dLast_err,c1000_err,c2000_err,c4000_err,c6000_err]
    error_data = [normals_err]
    # domain = np.linspace(1, 87, 87) # for non normals plots
    domain = np.linspace(1, 100, 100)
    # domain = list(range(1, 101))  # Domain from 0 to 100
    #labels = ['high','first','last','1 kHz','2 kHz','4 kHz','6 kHz']
    labels = ['high']
    markers = ['o','d']


    # plotting each scatterplot
    for data, label, color, marker in zip(scatter_data, labels, colours, markers):
        scatter_ax.scatter(domain[:87], data[:87], label=label, color=color, marker=marker, s=100, linestyle='')

    for data, err, color, marker in zip(scatter_data, error_data,colours, markers):
        scatter_ax.errorbar(domain[:87], data[:87], yerr=err[:87], capsize=5, color=color, elinewidth=0.75, marker=marker, linestyle='')
        
    # Making the last points red
    red_points = normals[-13:]
    x = [88,89,90,91,92,93,94,95,96,97,98,99,100]
    scatter_ax.scatter(x, red_points, color='red', label='low', marker='d', s=100)
    scatter_ax.errorbar(domain[-13:], normals[-13:], yerr=normals_err[-13:], fmt='d', capsize=5, color='red', elinewidth=0.75)

    # Plotting the grey lines
    for i in range(1, len(normals_words)+1):
        plt.vlines(i, 0, 100, colors='gray', linestyles='dashed', alpha=0.5)
    # # plt.axhline(y=0.8, color='red', linestyle='--', label='Accuracy Bound')  # Add the red dotted line at y=0.8

    # Format
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(10))
    # plt.legend(loc='lower left',fontsize=24, markersize=10)

    legend_handle_1 = Line2D([], [], marker='o', markersize=10, label=labels[0], color=colours[0])
    # legend_handle_2 = Line2D([], [], marker='d', markersize=10, label=labels[1], color=colours[1])  # Customize markersize
    legend_handle_3 = Line2D([], [], marker='d', markersize=10, label='low', color='red')  # Customize markersize
    plt.legend(handles=[legend_handle_1,legend_handle_3], bbox_to_anchor=(1, 1.15), loc='upper right', fontsize=18, ncol=3, title='classifier accuracy:', title_fontproperties={'weight': 'bold', 'size': 18})
    #plt.legend(handles=[legend_handle_1,legend_handle_3], loc='upper right', fontsize=12, ncol=2)
    scatter_ax.annotate("a)", xy=(0, 1.05), xycoords='axes fraction', fontsize=26, fontweight='bold')

    normals_words = list(normals_words)

    for i in range(len(normals_words)):
        if i % 2 == 1:
            normals_words[i] += ' ---------'

    plt.xticks(domain, normals_words, rotation=90, fontsize=16)
    plt.gca().yaxis.set_major_locator(MultipleLocator(10))
    plt.yticks(fontsize=18)  # Adjust the fontsize as needed
    # plt.title('Accuracy By Word', fontweight='bold',fontsize=26)
    plt.xlabel('PBK Word Stimuli - Unaltered',fontweight='bold', fontsize=26)
    plt.ylabel('Classifier Identification: Correct (%)', fontweight='bold', fontsize=26)
    plt.ylim(0,100)

    # Create the bar plot
    kde_ax = plt.subplot(gs[1])
    #values = [0.89, 0.99, 0.74, 0.86, 0.43, 0.21, 0.20]
    
    y_kde = np.arange(0, 101, 10)

    kde_normals = gaussian_kde(normals)
    density_normals = kde_normals(y_kde)
    kde_ax.plot(density_normals, y_kde, color='#1f77b4', label='high', linewidth=2.5)
    # Add the second plot with the red color
    kde_normals_altered = gaussian_kde(red_points)
    density_normals_altered = kde_normals_altered(y_kde)
    kde_ax.plot(density_normals_altered, y_kde, color='red', label='low', linestyle='dotted', linewidth=2.5)

    # kde_dFirst = gaussian_kde(dFirst)
    # density_dFirst = kde_dFirst(y_kde)
    # kde_dLast = gaussian_kde(dLast)
    # density_dLast = kde_dLast(y_kde)
    
    # kde_ax.plot(density_dFirst, y_kde, color='#ff7f0e', label='first', linewidth=2.5)
    # kde_ax.plot(density_dLast, y_kde, color='#2ca02c', label='last', linestyle='dotted', linewidth=2.5)

    # kde_c1000 = gaussian_kde(c1000)
    # density_c1000 = kde_c1000(y_kde)

    # kde_c2000 = gaussian_kde(c2000)
    # density_c2000 = kde_c2000(y_kde)

    # kde_c4000 = gaussian_kde(c4000)
    # density_c4000 = kde_c4000(y_kde)

    # kde_c6000 = gaussian_kde(c6000)
    # density_c6000 = kde_c6000(y_kde)

    # kde_ax.plot(density_c1000, y_kde, color='#d62728', label='1 kHz', linewidth=2.5)
    # kde_ax.plot(density_c2000, y_kde, color='#9467bd', label='2 kHz', linestyle='dotted', linewidth=2.5)
    # kde_ax.plot(density_c4000, y_kde, color='#8c564b', label='4 kHz', linewidth=2.5)
    # kde_ax.plot(density_c6000, y_kde, color='#e377c2', label='6 kHz', linestyle='dotted', linewidth=2.5)

    kde_ax.set_xlabel('Probability Density', fontweight='bold', fontsize=26, labelpad=15)
    kde_ax.yaxis.set_major_locator(MultipleLocator(10))
    kde_ax.yaxis.set_tick_params(labelsize=18)
    kde_ax.xaxis.set_tick_params(labelsize=18)
    # kde_ax.set_title('Kernel Density Estimate',fontweight='bold',fontsize=24)
    kde_ax.legend(loc='upper right',fontsize=18, bbox_to_anchor=(1, 1.15), ncol=2, title='classifier accuracy:', title_fontproperties={'weight': 'bold', 'size': 18})
    #kde_ax.legend(loc='upper right',fontsize=12, ncol=2)
    kde_ax.set_ylim(0, 100)  # Set y-axis limits from 0 to 100
    kde_ax.annotate("b)", xy=(0, 1.05), xycoords='axes fraction', fontsize=26, fontweight='bold')
    kde_ax.set_xticks([])

    #legend_labels = ['normals','dFirst','dLast','c1000','c2000','c4000','c6000']

    #plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.1))
    plt.tight_layout()  
    plt.show()

    

def dprime(hits, misses, correct_rejections, false_alarms):
    hit_rate = hits / (hits + misses)
    false_alarm_rate = false_alarms / (false_alarms + correct_rejections)

    z_hit = norm.ppf(hit_rate)
    z_false_alarm = norm.ppf(false_alarm_rate)

    d_prime = z_hit - z_false_alarm
    # return [d_prime, z_hit, z_false_alarm]

    if d_prime == float('inf'):
        return 5
    return d_prime

def d_prime_hist():
    legend_labels = ['Unaltered Stimuli','First Phoneme','Last Phoneme','c1000','c2000','c4000','c6000']
    colours = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    line_type = ['-','--',':','-.']

    hits = 1084
    misses = 116

    correct_rejections = [1187,921,1037,512,248,235]
    false_alarms = [13,279,161,688,952,965]

    hit_rate = hits / (hits + misses)
    z_hit = norm.ppf(hit_rate)

    # Calculate d' value
    stats = []
    for i in range (6):
        values = dprime(hits, misses, correct_rejections[i], false_alarms[i])
        rounded = [round(num, 3) for num in values]
        stats.append(rounded)

    # Generate data points for the normal distributions
    x = np.linspace(-4, 4, 1000)
    
    hit_distribution = norm.pdf(x, loc=z_hit, scale=1)

    dFirst_distribution = norm.pdf(x, loc=stats[0][2], scale=1)
    dLast_distribution = norm.pdf(x, loc=stats[1][2], scale=1)
    c1000_distribution = norm.pdf(x, loc=stats[2][2], scale=1)
    c2000_distribution = norm.pdf(x, loc=stats[3][2], scale=1)
    c4000_distribution = norm.pdf(x, loc=stats[4][2], scale=1)
    c6000_distribution = norm.pdf(x, loc=stats[5][2], scale=1)

    # Plot the normal distributions
    plt.figure(figsize=(14, 10))
    plt.plot(x, hit_distribution, label=f'{legend_labels[0]}', color=colours[0], linewidth=4.5, linestyle=line_type[0])

    plt.plot(x, dFirst_distribution, label=f"{legend_labels[1]}, d' = {stats[0][0]}", color=colours[1], linestyle=line_type[1], linewidth=2.5, marker='*', markevery=40, markersize=15)
    plt.plot(x, dLast_distribution, label=f"{legend_labels[2]}, d' = {stats[1][0]}", color=colours[2], linestyle=line_type[2], linewidth=2.5, marker='X', markevery=45, markersize=15)
    plt.plot(x, c1000_distribution, label=f"{legend_labels[3]}, d' = {stats[2][0]}", color=colours[3], linestyle=line_type[3], linewidth=2.5, marker='h', markevery=50, markersize=15)
    plt.plot(x, c2000_distribution, label=f"{legend_labels[4]}, d' = {stats[3][0]}", color=colours[4], linestyle=line_type[0], linewidth=2.5, marker='o', markevery=40, markersize=15)
    plt.plot(x, c4000_distribution, label=f"{legend_labels[5]}, d' = {stats[4][0]}", color=colours[5], linestyle=line_type[1], linewidth=2.5, marker='d', markevery=45, markersize=15)
    plt.plot(x, c6000_distribution, label=f"{legend_labels[6]}, d' = {stats[5][0]}", color=colours[6], linestyle=line_type[2], linewidth=2.5, marker='s', markevery=55, markersize=15)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)  # Adjust the fontsize as needed
    plt.xlabel('Z-Score',fontweight='bold',fontsize=30)
    plt.ylabel('Probability Density',fontweight='bold', fontsize=30)
    # plt.title('Comparison of Signal Detection Performance Across Conditions',fontweight='bold', fontsize=19)
    plt.legend(loc='upper right', fontsize=20)
    plt.grid(True)
    plt.tight_layout() 
    plt.show()

def intReliability_1():
    data = [
        [90, 99, 77, 87, 43, 21, 20],
        [88, 88, 56, 86, 48, 14, 18],
        [78, 94, 68, 94, 56, 28, 26],
        [76, 88, 64, 84, 42, 24, 12],
        [80, 80, 66, 82, 42, 12, 20],
        [88, 74, 74, 92, 36, 18, 10],
        [84, 82, 74, 88, 50, 20, 12],
        [82, 70, 44, 86, 38, 14, 16]
    ]
    
    labels = ["ursa",'human scorer 1','human scorer 2','human scorer 3','human scorer 4','human scorer 5','human scorer 6','human scorer 7']
    conditions =['Unaltered', 'First Phoneme \nDeletion', 'Last Phoneme \nDeletion', 'LFC 1 kHz', 'LFC 2 kHz', 'LFC 4 kHz', 'LFC 6 kHz']
    # colors = ['blue','#6CBDE9','#6CBDE9','#6CBDE9','#6CBDE9','#6CBDE9','#6CBDE9','#6CBDE9']
    # colors = ['white','white','white','white','white','white','white','white']
    num_bars = 8
    grey_values = [str(i / (num_bars + 1)) for i in range(1, num_bars + 1)]
    colors = grey_values

    w = 0.11  # Width of each bar

    plt.figure(figsize=(14, 9))

    x = np.arange(len(conditions))  # x-axis values for each condition

    #fill_patterns = ['/', '\\', 'o', 'x', '++', '-', '*', 'oo', '.']

    # for i, (row, label, color) in enumerate(zip(data, labels, colors)):
    #     plt.bar(x + (i * w), row, width=w, label=label, hatch=fill_patterns[i % len(fill_patterns)], edgecolor='black', linewidth=0.8, color=color)
    for i, (row, label, color) in enumerate(zip(data, labels, colors)):
        plt.bar(x + (i * w), row, width=w, label=label, edgecolor='black', linewidth=0.8, color=color)

    # Adjust layout and show the plot
    plt.annotate("a)", xy=(0, 1.05), xycoords='axes fraction', fontsize=26, fontweight='bold')
    plt.xlabel('Condition', fontweight='bold', fontsize=28)
    plt.ylabel('Accuracy (%)', fontweight='bold', fontsize=28)
    plt.xticks(x + (w * len(labels) / 2), conditions, fontsize=22, rotation=45)
    y_ticks = np.arange(0, 110, 10)
    plt.yticks(y_ticks,fontsize=22)
    plt.legend(prop={'size': 20})  # Increase the size of the legend
    plt.tight_layout()
    plt.show()

def intReliability_2():
    data = [
        [90, 88, 78, 76, 80, 88, 84, 82],
        [99, 88, 94, 88, 80, 74, 82, 70],
        [77, 56, 68, 64, 66, 74, 74, 44],
        [87, 86, 94, 84, 82, 92, 88, 86],
        [43, 48, 56, 42, 42, 36, 50, 38],
        [21, 14, 28, 24, 12, 18, 20, 14],
        [20, 18, 26, 12, 20, 10, 12, 16]
    ]

    labels = ["ursa", 'human scorer 1', 'human scorer 2', 'human scorer 3', 'human scorer 4', 'human scorer 5',
              'human scorer 6', 'human scorer 7']
    conditions = ['Unaltered', 'First Phoneme \nDeletion', 'Last Phoneme \nDeletion', 'LFC 1 kHz', 'LFC 2 kHz',
                  'LFC 4 kHz', 'LFC 6 kHz']

    w = 0.11  # Width of each bar

    plt.figure(figsize=(14, 9))

    x = np.arange(len(labels))  # x-axis values for each scorer

    for i, (row, condition) in enumerate(zip(data, conditions)):
        plt.bar(x + (i * w), row, width=w, label=condition, edgecolor='black', linewidth=0.8)

    # Adjust layout and show the plot
    plt.annotate("a)", xy=(0, 1.05), xycoords='axes fraction', fontsize=26, fontweight='bold')
    plt.xlabel('Scorer', fontweight='bold', fontsize=28)
    plt.ylabel('Accuracy (%)', fontweight='bold', fontsize=28)
    plt.xticks(x + (w * len(conditions) / 2), labels, fontsize=22, rotation=45)
    y_ticks = np.arange(0, 110, 10)
    plt.yticks(y_ticks, fontsize=22)
    plt.legend(prop={'size': 20}, title='Conditions')  # Include legend with condition titles
    plt.tight_layout()
    plt.show()

def intReliability_3():
    emm_accuracy = [83.3, 84.3, 65.4, 87.4, 44.3, 18.9, 16.7]
    emm_error_values = [1.77, 3.19, 4.03, 1.62, 2.68, 2.21, 2.11] # calculated
    # emm_error_values = [2.13, 2.13, 2.13, 2.13, 2.13, 2.13, 2.13] # From Emmeans
    markers = ['+', 's', 'D', '^', 'v', 'p', '*', 'H', 'X', '<', '>', 'x']
    # Rows represent people, coloumns represent conditions
    accuracy_by_person = [
        [88, 88, 56, 86, 48, 14, 18],
        [78, 94, 68, 94, 56, 28, 26],
        [76, 88, 64, 84, 42, 24, 12],
        [80, 80, 66, 82, 42, 12, 20],
        [88, 74, 74, 92, 36, 18, 10],
        [84, 82, 74, 88, 50, 20, 12],
        [82, 70, 44, 86, 38, 14, 16]
    ]
    ursa = [90, 99, 77, 87, 43, 21, 20]

    # Calculate standard errors for each condition
    # standard_errors = np.std(accuracy_by_person, axis=0, ddof=1) / np.sqrt(len(accuracy_by_person))

    # Print the standard errors for each condition
    # for condition, se in zip(range(1, standard_errors.size + 1), standard_errors):
    #     print(f"Condition {condition}: {se:.2f}")

    labels = ['Unaltered \nStimuli','First Phoneme \nDeletion','Last Phoneme \nDeletion','LFC 1 kHz','LFC 2 kHz','LFC 4 kHz','LFC 6 kHz']

    # Function to add a brace between two points
    def add_brace(ax, x1, x2, y, label=None):
        height = 5
        ax.annotate('', xy=(x1, y), xycoords='data',
                    xytext=(x1, y-height), textcoords='data',
                    arrowprops=dict(arrowstyle='-'))
        ax.annotate('', xy=(x2, y), xycoords='data',
                    xytext=(x2, y-height), textcoords='data',
                    arrowprops=dict(arrowstyle='-'))
        ax.annotate('', xy=(x1, y-height), xycoords='data',
                    xytext=(x2, y-height), textcoords='data',
                    arrowprops=dict(arrowstyle='-'))
        if label:
            ax.text((x1+x2)/2., y-height-0.1, label, ha='center', va='top', fontweight='bold', fontsize=12)

    plt.figure(figsize=(14, 9))
    domain = np.arange(1,8)
    offset = 0.04  # Adjust this value to control the horizontal offset
    has_label_added = False

    add_brace(plt, 1, 3, 25, label='*')  # SD: Unaltered to last
    add_brace(plt, 2, 3, 15, label='*')  # SD: first to last
    add_brace(plt, 4, 5, 25, label='*')  # SD: 1 to 2
    add_brace(plt, 5, 6, 15, label='*')  # SD: first to last

    for i, (person) in enumerate(zip(accuracy_by_person)):
        point_offset = (-1)**i * offset * (i//2)
        if not has_label_added:
            plt.scatter(domain + point_offset, person, label='human \nscorers', color='#6CBDE9', s=140, marker='X', linestyle='')
            has_label_added = True
        else:
            plt.scatter(domain + point_offset, person, color='#6CBDE9', s=140, marker='X', linestyle='')


    plt.errorbar(domain, emm_accuracy, yerr=emm_error_values, label='\nestimated \nmarginal \nmean of \nhuman \nscorers', markersize=12, color='blue', marker='o', linestyle='', capsize=5, elinewidth=2)
    plt.scatter(domain-0.03, ursa, label='\nmachine \nclassifier', s=120, color='orange', marker='D', linestyle='')

    #plt.annotate("a)", xy=(0, 1.05), xycoords='axes fraction', fontsize=26, fontweight='bold')

    plt.grid(True, axis='y')
    plt.xticks(domain, labels, rotation=45, fontsize=20)
    y_ticks = np.arange(0, 110, 10)
    plt.yticks(y_ticks,fontsize=20)
    plt.xlabel('Condition',fontweight='bold', fontsize=26)
    plt.ylabel('Accuracy (%)', fontweight='bold', fontsize=26)
    plt.ylim(0,100)
    plt.legend(prop={'size': 20}, loc='center left', bbox_to_anchor=(1, 0.5))  # Place legend to the right of the plot
    plt.subplots_adjust(bottom=0.3, right=0.75)  # Adjust the right margin to fit the legend
    plt.show()

def intReliability_4():
    emm_accuracy = [62.4, 56.9, 63.4, 55.7, 54.6, 56.0, 58.6, 50]
    # emm_error_values = [2.34, 2.34, 2.34, 2.34, 2.34, 2.34, 2.34, 2.34] # from Emmeans
    emm_error_values = [12.73, 12.16, 10.71, 11.37, 11.28, 12.84, 11.98, 11.29] # From calculations

    markers = ['o', 's', 'D', '^', 'v', 'p', '*', 'H']
    condition_labels = ['unaltered stimuli', 'first phoneme \ndeletion', 'last phoneme \ndeletion', 'LFC 1 kHz', 'LFC 2kHz', 'LFC 4 kHz', 'LFC 6 kHz']

    accuracy_by_person = [
        [90, 99, 77, 87, 43, 21, 20],
        [88, 88, 56, 86, 48, 14, 18],
        [78, 94, 68, 94, 56, 28, 26],
        [76, 88, 64, 84, 42, 24, 12],
        [80, 80, 66, 82, 42, 12, 20],
        [88, 74, 74, 92, 36, 18, 10],
        [84, 82, 74, 88, 50, 20, 12],
        [82, 70, 44, 86, 38, 14, 16]
    ]

    person_labels = ['Ursa', 'Human\nScorer 1', 'Human\nScorer 2', 'Human\nScorer 3', 'Human\nScorer 4', 'Human\nScorer 5', 'Human\nScorer 6', 'Human\nScorer 7']

    plt.figure(figsize=(14, 9))
    domain = np.arange(1, 9)
    offset = np.linspace(-0.3, 0.3, 7)  # Create 7 offsets for the 7 conditions

    # Transpose the accuracy_by_person list for proper iteration
    transposed_accuracies = list(zip(*accuracy_by_person))

    # # Calculate standard errors for each condition
    # standard_errors = np.std(transposed_accuracies, axis=0, ddof=1) / np.sqrt(len(transposed_accuracies))

    # ### Print the standard errors for each condition
    # for condition, se in zip(range(1, standard_errors.size + 1), standard_errors):
    #     print(f"Condition {condition}: {se:.2f}")

    for i, (condition, marker) in enumerate(zip(transposed_accuracies, markers)):
        points_x = domain + offset[i]  # Apply offset to each condition
        plt.scatter(points_x, condition, color='#6CBDE9', s=140, marker=marker, linestyle='', label=condition_labels[i])

    plt.errorbar(domain, emm_accuracy, yerr=emm_error_values, label='estimated \nmarginal mean', markersize=12, color='blue', marker='o', linestyle='', capsize=5, elinewidth=2)

    plt.annotate("b)", xy=(0, 1.05), xycoords='axes fraction', fontsize=26, fontweight='bold')

    plt.grid(True, axis='y')
    plt.xticks(domain, person_labels, rotation=45, fontsize=20)
    y_ticks = np.arange(0, 110, 10)
    plt.yticks(y_ticks, fontsize=20)
    plt.xlabel('Scorer', fontweight='bold', fontsize=26)
    plt.ylabel('Accuracy (%)', fontweight='bold', fontsize=26)
    plt.ylim(0, 100)
    plt.subplots_adjust(bottom=0.3, right=0.75)  # Adjust the right margin to fit the legend
    plt.legend(prop={'size': 20}, loc='center left', bbox_to_anchor=(1, 0.5))  # Place legend to the right of the plot
    plt.show()

def lmmDprimePlot():
    data = []
    #people = ['aCauchi','Claire','Cole','Daniel','Hillary','Lulia','Melissa','Micheal','Polonenko','rCauchi','Robel','Samsoor']
    people = ['aCauchi','Cole','Daniel','Micheal','Robel','Samsoor','Claire','Hillary','Lulia','Melissa','Polonenko','rCauchi']
    ids = ['ID: 30385','ID: 24535','ID: 29501','ID: 32648','ID: 69215','ID: 93459','ID: 40124','ID: 82784','ID: 95370','ID: 56489','ID: 96742','ID: 14539']
    labels = ['First Phoneme','Last Phoneme','c1000','c2000','c4000','c6000']
    colours = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#1f77b4', '#ff9896']
    markers = ['o', 's', 'D', '^', 'v', 'p', '*', 'H', 'X', '<', '>', 'x']

    for person, id in zip(people, ids):
        info = pullDPrimeMetricsByPerson(person)
        c_rej = info['other_data']['c_rej']
        f_alarm = info['other_data']['f_alarm']

        d_primes = [
            round(dprime(info['hit'],info['miss'],c_rej['dFirst'],f_alarm['dFirst']),2),            
            round(dprime(info['hit'],info['miss'],c_rej['c1000'],f_alarm['c1000']),2),
            round(dprime(info['hit'],info['miss'],c_rej['dLast'],f_alarm['dLast']),2),
            round(dprime(info['hit'],info['miss'],c_rej['c2000'],f_alarm['c2000']),2),
            round(dprime(info['hit'],info['miss'],c_rej['c4000'],f_alarm['c4000']),2),
            round(dprime(info['hit'],info['miss'],c_rej['c6000'],f_alarm['c6000']),2),
        ]

        person_data = {
            "name": person,
            "id": id,
            "d_primes": d_primes,
        }
        data.append(person_data)

    names = [segment['name'] for segment in data]
    ids = [segment['id'] for segment in data]
    d_primes = [segment['d_primes'] for segment in data]

    #######################
    # Trying to make this work in R
    # names_r = ro.StrVector(names)
    # ids_r = ro.StrVector(ids)
    # d_primes_r = ro.r.matrix(ro.FloatVector([val for sublist in d_primes for val in sublist]), nrow=len(d_primes), byrow=True)

    # # Assign these R objects to the R environment
    # ro.globalenv['names_var'] = names_r
    # ro.globalenv['ids_var'] = ids_r
    # ro.globalenv['d_primes_var'] = d_primes_r

    # base = importr('base')
    # grdevices = importr('grDevices')
    
    # # Load your R script file
    # script_path = "analysis/plotting.R"
    # with open(script_path, "r") as r_script:
    #     r_code = r_script.read()

    # # Evaluate the R code
    # r(r_code)
    # r("lmmDprimePlot(names_var, ids_var, d_primes_var)")
    #######################

    plt.figure(figsize=(5, 3))
    domain = np.arange(1,7)
    for values, name, id, colour, marker in zip(d_primes, names, ids, colours, markers):
        plt.scatter(domain, values, label=id, color=colour, s=100, marker=marker, linestyle='')

    # Plotting the base mean for each condition
    means, std_error = [], []
    reversed_array = [[d_primes[j][i] for j in range(len(d_primes))] for i in range(len(d_primes[0]))]
    for subarray in reversed_array:
        means.append(sum(subarray)/len(subarray))
        std_error.append((np.std(subarray, ddof=1))/np.sqrt(len(subarray)))    
    plt.errorbar(domain, means, yerr=std_error, label='Base Mean ± 1 SE', markersize=12, color='orange', marker='o', linestyle='', capsize=5)


    # Create a Pandas DataFrame for the analysis
    df = pd.DataFrame({'Condition': labels * len(data), 'D_Prime': np.array(d_primes).flatten()})

    # Fit an OLS regression model treating "Condition" as categorical
    model = sm.OLS.from_formula("D_Prime ~ C(Condition)", data=df).fit()

    # Group the data by 'Condition' and calculate means for each group
    emms = df.groupby('Condition')['D_Prime'].mean().values

    # Plot the estimated marginal means (EMMs)
    plt.plot(domain, emms, marker='o', color='red', markersize=12, label="EMM", linestyle='')

    plt.xticks(domain, labels, rotation=45, fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Condition',fontweight='bold', fontsize=26)
    plt.ylabel('D-Prime', fontweight='bold', fontsize=26)
    plt.ylim(-0.5,5.5)
    plt.legend(loc='upper right',fontsize=16, ncol=3)
    plt.subplots_adjust(bottom=0.3)  # Increase bottom margin as needed
    plt.show()

def dPrime_plot():
    one = [4.11,2.73,2.1,0.93,0.58,0.65]
    two = [3.03,2.11,1.58,0.66,0.0,0.03]
    three = [5,3.63,3.11,2.32,2.19,2.19]
    four = [5,2.76,2.76,1.14,0.19,0.35]
    five = [5,2.14,1.87,0.87,0.18,0.24]
    six = [5,2.0,1.87,0.45,0,0]
    seven = [2.9,2.75,1.29,1.47,0.6,0.5]
    eight = [5,2.75,3.09,1.68,0.8,0.58]
    nine = [3.61,2.51,1.69,1.16,0.4,0.25]
    ten = [5,2.29,2.36,1.34,0.51,0.34]
    eleven = [4.38,2.9,2.76,1.59,0.58,0.5]
    twelve = [3.39,2.29,2.15,0.9,0.21,0.26]

    data = [one,two,three,four,five,six,seven,eight,nine,ten,eleven,twelve]

    # Calculate standard errors for each condition
    standard_errors = np.std(data, axis=0, ddof=1) / np.sqrt(len(data))

    # Print the standard errors for each condition
    for condition, se in zip(range(1, standard_errors.size + 1), standard_errors):
        print(f"Condition {condition}: {se:.2f}")

    #colours = ['#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#1f77b4', '#ff9896']
    #markers = ['+', 's', 'D', '^', 'v', 'p', '*', 'H', 'X', '<', '>', 'x']
    #study_ID = ['participant 1','participant 2','participant 3','participant 4','participant 5','participant 6','participant 7','participant 8','participant 9','participant 10','participant 11','participant 12']
    standard_err = [0.24,0.13,0.17,0.15,0.17,0.16]
    emmeans = [4.285,2.5717,2.21917,1.20917,0.52,0.49083]
    labels = ['First Phoneme \nDeletion','Last Phoneme \nDeletion','LFC 1 kHz','LFC 2 kHz','LFC 4 kHz','LFC 6 kHz']

    plt.figure(figsize=(14, 9))
    domain = np.arange(1,7)
    offset = 0.02  # Adjust this value to control the horizontal offset
    plt.errorbar(domain, emmeans, yerr=standard_err, label="\nestimated \nmarginal mean \nof d' over \nspeakers\n", markersize=12, color='blue', marker='o', linestyle='', capsize=5, elinewidth=2)
    #for i, (person, id, marker, colour) in enumerate(zip(data, study_ID, markers, colours)):
    has_label_added = False  # This ensures we only add the label once
    for i, person in enumerate(data):
        # Alternate offset direction for each point
        point_offset = (-1)**i * offset * (i//2)
        if not has_label_added:
            plt.scatter(domain + point_offset, person, label='machine signal \n(unaltered) in \nnoise (altered) \ndiscriminability \nper speaker', color='#6CBDE9', s=140, marker='X', linestyle='')
            has_label_added = True
        else:
            plt.scatter(domain + point_offset, person, color='#6CBDE9', s=140, marker='X', linestyle='')

    # Function to add a brace between two points
    def add_brace(ax, x1, x2, y, label=None):
        height = 0.25
        ax.annotate('', xy=(x1, y), xycoords='data',
                    xytext=(x1, y-height), textcoords='data',
                    arrowprops=dict(arrowstyle='-'))
        ax.annotate('', xy=(x2, y), xycoords='data',
                    xytext=(x2, y-height), textcoords='data',
                    arrowprops=dict(arrowstyle='-'))
        ax.annotate('', xy=(x1, y-height), xycoords='data',
                    xytext=(x2, y-height), textcoords='data',
                    arrowprops=dict(arrowstyle='-'))
        if label:
            ax.text((x1+x2)/2., y-height-0.1, label, ha='center', va='top', fontweight='bold', fontsize=12)
    
    # Add braces
    add_brace(plt, 1, 2, 0.25, label='*')  # SD: first to last phoneme deletion
    add_brace(plt, 3, 4, 0.25, label='*')  # SD: 1 to 2 kHz 
    add_brace(plt, 4, 5, 0.25, label='*')  # SD: 1 to 2 kHz 

    plt.grid(True, axis='y')
    plt.xticks(domain, labels, rotation=45, fontsize=20)
    y_ticks = np.arange(0, 5.5, .5)
    plt.yticks(y_ticks,fontsize=20)
    plt.xlabel('Condition',fontweight='bold', fontsize=26)
    plt.ylabel("d'", fontweight='bold', fontsize=26)
    plt.ylim(-0.5,5.5)
    # plt.legend(loc='upper right',fontsize=20, ncol=1)
    plt.subplots_adjust(bottom=0.3, right=0.75)  # Increase bottom and right margin as needed
    plt.legend(prop={'size': 20}, loc='center left', bbox_to_anchor=(1, 0.5))  # Place legend to the right of the plot
    plt.show()

def dPrime_plot_2():
    one = [4.11,2.73,2.1,0.93,0.58,0.65]
    two = [3.03,2.11,1.58,0.66,0.0,0.03]
    three = [5,3.63,3.11,2.32,2.19,2.19]
    four = [5,2.76,2.76,1.14,0.19,0.35]
    five = [5,2.14,1.87,0.87,0.18,0.24]
    six = [5,2.0,1.87,0.45,0,0]
    seven = [2.9,2.75,1.29,1.47,0.6,0.5]
    eight = [5,2.75,3.09,1.68,0.8,0.58]
    nine = [3.61,2.51,1.69,1.16,0.4,0.25]
    ten = [5,2.29,2.36,1.34,0.51,0.34]
    eleven = [4.38,2.9,2.76,1.59,0.58,0.5]
    twelve = [3.39,2.29,2.15,0.9,0.21,0.26]
    data_ml = [one,two,three,four,five,six,seven,eight,nine,ten,eleven,twelve]

    Cole = [2.26, 0.93, 2.16, 0.77, 0.0, 0.09]
    Joyce = [2.73, 1.82, 2.35, 1.17, 0.4, 0.0]
    Ryley = [2.58, 2.05, 2.81, 1.05, 0.49, 0.12]
    Lulia = [1.84, 1.41, 1.91, 0.79, 0, 0.15]
    Isabel = [1.48, 1.2, 1.84, 0.64, 0.14, 0]
    Susan = [1.83, 1.38, 2.47, 1.07, 0.33, 0.27]
    Maria = [1.93, 1.56, 2.49, 1.35, 0.32, 0.49]
    data_people = [Cole,Joyce,Ryley,Lulia,Isabel,Susan,Maria]

    # Means for ML and People
    emmeans_ml = [4.285, 2.5717, 2.21917, 1.20917, 0.52, 0.49083]
    emmeans_people = [np.mean([person[i] for person in data_people]) for i in range(6)]

    # Error bars for ML and People
    standard_err_ml = [0.24, 0.13, 0.17, 0.15, 0.17, 0.16]
    standard_err_people = [np.std([person[i] for person in data_people], ddof=1) / np.sqrt(len(data_people)) for i in range(6)]

    labels = ['First Phoneme \nDeletion','Last Phoneme \nDeletion','LFC 1 kHz','LFC 2 kHz','LFC 4 kHz','LFC 6 kHz']

    plt.figure(figsize=(14, 9))
    domain = np.arange(1,7)
    offset = 0.04  # Adjust this value to control the horizontal offset

    plt.errorbar(domain, emmeans_ml, yerr=standard_err_ml, markersize=15, color='blue', marker='o', linestyle='', capsize=5, elinewidth=2)
    plt.errorbar(domain, emmeans_people, yerr=standard_err_people, markersize=15, color='#8B0000', marker='o', linestyle='', capsize=5, elinewidth=2)

    has_label_added = False  # This ensures we only add the label once
    for i, person in enumerate(data_ml):
        # Alternate offset direction for each point
        point_offset = (-1)**i * offset * (i//2)
        if not has_label_added:
            plt.scatter(domain + point_offset, person, label='unique\nmachine scorer\ndiscriminability\nover varrying\nspeakers\n', color='#6CBDE9', s=70, marker='o', linestyle='')
            has_label_added = True
        else:
            plt.scatter(domain + point_offset, person, color='#6CBDE9', s=70, marker='o', linestyle='')

    has_label_added = False  # This ensures we only add the label once
    for i, person in enumerate(data_people):
        # Alternate offset direction for each point
        point_offset = (-1)**i * offset * (i//2)
        if not has_label_added:
            plt.scatter(domain + point_offset, person, label='various\nhuman scorers\ndiscriminability\nover unique\nstimuli', color='red', s=70, marker='o', linestyle='')
            has_label_added = True
        else:
            plt.scatter(domain + point_offset, person, color='red', s=70, marker='o', linestyle='')

    # Function to add a brace between two points
    def add_brace(ax, x1, x2, y, label=None):
        height = 0.25
        ax.annotate('', xy=(x1, y), xycoords='data',
                    xytext=(x1, y-height), textcoords='data',
                    arrowprops=dict(arrowstyle='-'))
        ax.annotate('', xy=(x2, y), xycoords='data',
                    xytext=(x2, y-height), textcoords='data',
                    arrowprops=dict(arrowstyle='-'))
        ax.annotate('', xy=(x1, y-height), xycoords='data',
                    xytext=(x2, y-height), textcoords='data',
                    arrowprops=dict(arrowstyle='-'))
        if label:
            ax.text((x1+x2)/2., y-height-0.1, label, ha='center', va='top', fontweight='bold', fontsize=12)
    
    # Add braces
    add_brace(plt, 1, 2, 0.25, label='*')  # SD: first to last phoneme deletion
    add_brace(plt, 3, 4, 0.25, label='*')  # SD: 1 to 2 kHz 
    add_brace(plt, 4, 5, 0.25, label='*')  # SD: 1 to 2 kHz 

    plt.grid(True, axis='y')
    plt.xticks(domain, labels, rotation=45, fontsize=20)
    y_ticks = np.arange(0, 5.5, .5)
    plt.yticks(y_ticks,fontsize=20)
    plt.xlabel('Condition',fontweight='bold', fontsize=26)
    plt.ylabel("d'", fontweight='bold', fontsize=26)
    plt.ylim(-0.5,5.5)
    plt.subplots_adjust(bottom=0.3, right=0.75)  # Increase bottom and right margin as needed
    plt.legend(prop={'size': 20}, loc='center left', bbox_to_anchor=(1, 0.5))  # Place legend to the right of the plot
    plt.show()

def agreement_plot():
    emm_data = [67.6, 80, 55.9, 81.7, 60.7, 66.4, 76.3]
    # yerr = [2.41, 2.41, 2.41, 2.41, 2.41, 2.41, 2.41] # from Emmeans
    yerr = [1.56, 3.35, 3.62, 0.81, 1.92, 3.01, 1.02] # from calculations

    study_ids = ['ursa to scorer 1','ursa to scorer 2','ursa to scorer 3','ursa to scorer 4','ursa to scorer 5','ursa to scorer 6','ursa to scorer 7']
    markers = ['+', 's', 'D', '^', 'v', 'p', '*', 'H', 'X', '<', '>', 'x']
    percent_agreement_by_person = [
        [67,67,47,82,61,65,74],
        [69,80,63,84,65,55,74],
        [71,71,55,78,67,59,76],
        [59,78,61,80,55,67,80],
        [67,86,61,82,63,78,74],
        [71,92,65,82,53,67,80],
        [69,86,39,84,61,74,76],
    ]

    # # Calculate standard errors for each condition
    # standard_errors = np.std(percent_agreement_by_person, axis=0, ddof=1) / np.sqrt(len(percent_agreement_by_person))

    # ### Print the standard errors for each condition
    # for condition, se in zip(range(1, standard_errors.size + 1), standard_errors):
    #     print(f"Condition {condition}: {se:.2f}")

    labels = ['Unaltered \nStimuli','First Phoneme \nDeletion','Last Phoneme \nDeletion','LFC 1 kHz','LFC 2 kHz','LFC 4 kHz','LFC 6 kHz']

    # Function to add a brace between two points
    def add_brace(ax, x1, x2, y, label=None):
        height = 5
        ax.annotate('', xy=(x1, y), xycoords='data',
                    xytext=(x1, y-height), textcoords='data',
                    arrowprops=dict(arrowstyle='-'))
        ax.annotate('', xy=(x2, y), xycoords='data',
                    xytext=(x2, y-height), textcoords='data',
                    arrowprops=dict(arrowstyle='-'))
        ax.annotate('', xy=(x1, y-height), xycoords='data',
                    xytext=(x2, y-height), textcoords='data',
                    arrowprops=dict(arrowstyle='-'))
        if label:
            ax.text((x1+x2)/2., y-height-0.1, label, ha='center', va='top', fontweight='bold', fontsize=12)

    plt.figure(figsize=(14, 9))
    domain = np.arange(1,8)
    offset = 0.035  # Adjust this value to control the horizontal offset
    has_label_added = False

    add_brace(plt, 1, 2, 35, label='*')  # SD: Unaltered to first
    add_brace(plt, 1, 3, 25, label='*')  # SD: Unaltered to last
    add_brace(plt, 2, 3, 15, label='*')  # SD: first to last

    add_brace(plt, 4, 5, 25, label='*')  # SD: 1 to 2
    add_brace(plt, 6, 7, 25, label='*')  # SD: 4 to 6

    #plt.scatter(domain, percent_agreement_p_vs_c, label='all participants with ursa', s=120, color='orange', marker='D', linestyle='')

    for i, (person) in enumerate(zip(percent_agreement_by_person)):
        point_offset = (-1)**i * offset * (i//2)
        if not has_label_added:
            plt.scatter(domain + point_offset, person, label='individual \nparticipant \nto ursa', color='#6CBDE9', s=140, marker='X', linestyle='')
            has_label_added = True
        else:
            plt.scatter(domain + point_offset, person, color='#6CBDE9', s=140, marker='X', linestyle='')


    plt.errorbar(domain, emm_data, yerr=yerr, label='estimated \nmarginal mean', markersize=12, color='blue', marker='o', linestyle='', capsize=5, elinewidth=2)
    # plt.scatter(domain+0.03, percent_agreement_all_p, label='all participants without ursa', s=160, color='orange', marker='*', linestyle='')

    plt.annotate("a)", xy=(0, 1.05), xycoords='axes fraction', fontsize=26, fontweight='bold')

    plt.grid(True, axis='y')
    plt.xticks(domain, labels, rotation=45, fontsize=20)
    y_ticks = np.arange(0, 110, 10)
    plt.yticks(y_ticks,fontsize=20)
    plt.xlabel('Condition',fontweight='bold', fontsize=26)
    plt.ylabel('Percent Agreement (%)', fontweight='bold', fontsize=26)
    plt.ylim(0,100)
    plt.legend(prop={'size': 20}, loc='center left', bbox_to_anchor=(1, 0.5))  # Place legend to the right of the plot
    plt.subplots_adjust(bottom=0.3, right=0.75)  # Adjust right margin to make room for legend
    plt.show()

def agreement_plot_2():
    emm_accuracy = [66.1, 70, 68.1, 68.6, 73, 72.9, 69.9]
    # emm_error_values = [2.41, 2.41, 2.41, 2.41, 2.41, 2.41, 2.41] # from emmeans
    emm_error_values = [4.11, 3.82, 3.21, 4.04, 3.64, 4.87, 6.07] # from calculations
    markers = ['s', 'D', '^', 'v', 'p', '*', 'H', 'X', 'x', '|']

    # Coloumns represent people, Rows represent conditions
    agreement_by_condition = [
        [67, 69, 71, 59, 67, 71, 69],
        [67, 80, 71, 78, 86, 92, 86],
        [47, 63, 55, 61, 61, 65, 39],
        [82, 84, 78, 80, 82, 82, 84],
        [61, 65, 67, 55, 63, 53, 61],
        [65, 55, 59, 67, 78, 67, 74],
        [74, 74, 76, 80, 74, 80, 76],
    ]

    # Calculate standard errors for each condition
    # standard_errors = np.std(agreement_by_condition, axis=0, ddof=1) / np.sqrt(len(agreement_by_condition))

    # # Print the standard errors for each condition
    # for condition, se in zip(range(1, standard_errors.size + 1), standard_errors):
    #     print(f"Condition {condition}: {se:.2f}")

    condition_labels = ['unaltered \nstimuli','first phoneme \ndeletion','last phoneme \ndeletion','LFC 1 kHz','LFC 2 kHz','LFC 4 kHz','LFC 6 kHz']
    person_labels = ['Ursa to Human\nScorer 1', 'Ursa to Human\nScorer 2', 'Ursa to Human\nScorer 3', 'Ursa to Human\nScorer 4', 'Ursa to Human\nScorer 5', 'Ursa to Human\nScorer 6', 'Ursa to Human\nScorer 7']

    plt.figure(figsize=(14, 9))
    domain = np.arange(1,8)
    offset = 0.05  # Adjust this value to control the horizontal offset
    has_label_added = False

    for i, (condition, condition_label, marker) in enumerate(zip(agreement_by_condition, condition_labels, markers)):
        point_offset = (-1)**i * offset * (i//2)
        if not has_label_added:
            plt.scatter(domain + point_offset, condition, label=condition_label, color='#6CBDE9', s=140, marker=marker, linestyle='')
            has_label_added = True
        else:
            plt.scatter(domain + point_offset, condition, label=condition_label, color='#6CBDE9', s=140, marker=marker, linestyle='')


    plt.errorbar(domain, emm_accuracy, yerr=emm_error_values, markersize=15, color='blue', marker='o', linestyle='', capsize=5, elinewidth=2)

    # plt.annotate("b)", xy=(0, 1.05), xycoords='axes fraction', fontsize=26, fontweight='bold')

    plt.grid(True, axis='y')
    plt.xticks(domain, person_labels, rotation=45, fontsize=20)
    y_ticks = np.arange(0, 110, 10)
    plt.yticks(y_ticks,fontsize=20)
    plt.xlabel('Scorer',fontweight='bold', fontsize=26)
    plt.ylabel('Percent Agreement (%)', fontweight='bold', fontsize=26)
    plt.ylim(0,100)
    plt.subplots_adjust(bottom=0.3, right=0.75)  # Increase bottom and right margin as needed
    plt.legend(prop={'size': 20}, loc='center left', bbox_to_anchor=(1, 0.5))  # Place legend to the right of the plot
    plt.show()
