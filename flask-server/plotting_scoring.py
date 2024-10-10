
#########################
# Imports:
from analysis.word_analysis import plotMeanAccByWord, d_prime_hist, intReliability_1, intReliability_2, intReliability_3, intReliability_4, lmmDprimePlot, dPrime_plot, dPrime_plot_2, agreement_plot, agreement_plot_2, dprime, segmentResponces
from analysis.ploting.ranking_normals import two_dimensional_grid, violinPlot, accuracyBarPlot,accVsCon

from analysis.accuracy_plots import extract_data, compare_word_lists, score_condition_by_person, accuracy_over_people, create_word_acc_std, std_error_over_people, create_acc_plot_unaltered, create_acc_plot_altered, create_boxplot, determine_P_N, determine_accuracy, determine_fscore, determine_precision, determine_recall, modify_scores, re_order_interrater, plot_metrics, get_evaluation_results, create_multiple_acc_plot_unaltered, create_multiple_acc_plot_altered, eval_human_for_big_plots, overall_accuracy_plot
import json
import numpy as np
#########################

# Running:

# list_number = 2
# kind = 'deletion'

# two_dimensional_grid('scatter',1,'data/json/segmented_norms/L1_normal/')
# two_dimensional_grid('violin',1,'data/json/segmented_norms/L1_normal/')
# two_dimensional_grid('violin',1,'data/json/segmented_norms/L1_deletion/')

# accuracyBarPlot(list_number, 'data/json/segmented_norms/L{}_{}/'.format(list_number,kind))

# plotMeanAccByWord()
# d_prime_hist()


# accVsCon(list_number, f'data/json/segmented_norms/L{list_number}_{kind}/')

# intReliability_1()
# intReliability_2()

# intReliability_3()
# intReliability_4()

# lmmDprimePlot()
# dPrime_plot()
# dPrime_plot_2()

# excel()
# percent_agreement()

# agreement_plot()
# agreement_plot_2()


### Good Looking PLots ###
# master_lists = ['speechmatics.json','whisper_large.json', 'whisper_medium.json'] 
# people = ['aCauchi', 'Claire', 'Cole', 'Daniel', 'Hillary', 'Lulia', 'Melissa', 'Micheal', 'Polonenko', 'rCauchi', 'Robel', 'Samsoor']
# conditions = ['dFirst', 'dLast', 'c1000', 'c2000', 'c4000', 'c6000']
# colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
# markers = ['o', 'd']

# for master in master_lists:

#     L1_normal, L2_normal = extract_data(master, condition='normal')
#     normal_scores = score_condition_by_person(L1_normal, L2_normal, people)

#     normal_dict = create_word_acc_std(
#         accuracy_scores_by_word=accuracy_over_people(normal_scores), 
#         standard_errors_by_word=std_error_over_people(normal_scores)
#     )

#     high_acc_words = create_acc_plot_unaltered(normal_dict)

#     for (condition_1, condition_2), (color_1, color_2) in zip(zip(conditions[0::2], conditions[1::2]), zip(colors[0::2], colors[1::2])):        
        
#         L1_c1, L2_c1 = extract_data(master, condition=condition_1)
#         c1_scores = score_condition_by_person(L1_c1, L2_c1, people)

#         L1_c2, L2_c2 = extract_data(master, condition=condition_2)
#         c2_scores = score_condition_by_person(L1_c2, L2_c2, people)

#         if condition_1 == 'dFirst' and condition_2 == 'dLast':
#             words_removed_dFirst = ['are', 'is', 'end', 'own', 'axe', 'all', 'as', 'on']
#             c1_scores = modify_scores(c1_scores, words_removed_dFirst)

#             words_removed_dLast = ['no', 'tree', 'lay', 'me', 'few', 'plow', 'gray', 'bee', 'grew', 'knee', 'tray']
#             c2_scores = modify_scores(c2_scores, words_removed_dLast)


#         c1_dict = create_word_acc_std(
#             accuracy_scores_by_word=accuracy_over_people(c1_scores), 
#             standard_errors_by_word=std_error_over_people(c1_scores)
#         )

#         c2_dict = create_word_acc_std(
#             accuracy_scores_by_word=accuracy_over_people(c2_scores), 
#             standard_errors_by_word=std_error_over_people(c2_scores)
#         )

#         create_acc_plot_altered(c1_dict, c2_dict, color_1, color_2, high_acc_words, condition_1, condition_2)

# get_fleis_kappa('whisper medium.json')







# ##### Good Looking Plots Multiple #####
# master_lists = ['ursa.json','whisper large.json', 'whisper medium.json'] 
master_lists = ['Whisper large.json', 'Whisper medium.json','Ursa.json'] 
# master_lists = ['deepspeech.json'] 
people = ['aCauchi', 'Claire', 'Cole', 'Daniel', 'Hillary', 'Lulia', 'Melissa', 'Micheal', 'Polonenko', 'rCauchi', 'Robel', 'Samsoor']
conditions = ['dFirst', 'dLast', 'c1000', 'c2000', 'c4000', 'c6000']
condition_names = ['First Consonant Deletion', 'Last Consonant Deletion', 'LPF 1 kHz', 'LPF 2 kHz', 'LPF 4 kHz', 'LPF 6 kHz']
# colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
colors = ['#203864', '#4472c4', '#00b0f0', '#ff0000']
markers = ['o', 'd']

condition_word_acc_dict = eval_human_for_big_plots()

master_normal = {}
for master in master_lists:

    L1_normal, L2_normal = extract_data(master, condition='normal')
    normal_scores = score_condition_by_person(L1_normal, L2_normal, people)

    normal_dict = create_word_acc_std(
        accuracy_scores_by_word=accuracy_over_people(normal_scores), 
        standard_errors_by_word=std_error_over_people(normal_scores),
        condition='normal'
    )
    master_normal[master] = normal_dict

master_normal['human'] = condition_word_acc_dict['Normal']
low_acc_words_dict, all_words = create_multiple_acc_plot_unaltered(master_normal, differentiate=False)

for condition, condition_name in zip(conditions, condition_names):

    master_alterations = {}
    for master in master_lists:

        L1_condition, L2_condition = extract_data(master, condition=condition)
        condition_scores = score_condition_by_person(L1_condition, L2_condition, people)

        condition_dict = create_word_acc_std(
            accuracy_scores_by_word=accuracy_over_people(condition_scores), 
            standard_errors_by_word=std_error_over_people(condition_scores),
            condition = condition
        )

        master_alterations[master] = condition_dict
    master_alterations["human"] = condition_word_acc_dict[condition]

    create_multiple_acc_plot_altered(master_alterations, low_acc_words_dict, all_words, condition_name)


# import matplotlib.pyplot as plt

# # Assuming you have all your imports and placeholder functions defined here

# master_lists = ['ursa.json', 'whisper large.json', 'whisper medium.json']
# people = ['aCauchi', 'Claire', 'Cole', 'Daniel', 'Hillary', 'Lulia', 'Melissa', 'Micheal', 'Polonenko', 'rCauchi', 'Robel', 'Samsoor']
# conditions = ['c1000', 'c2000', 'c4000', 'c6000']
# condition_names = ['LPF 1 kHz', 'LPF 2 kHz', 'LPF 4 kHz', 'LPF 6 kHz']

# condition_word_acc_dict = eval_human_for_big_plots()

# master_normal = {}
# for master in master_lists:
#     L1_normal, L2_normal = extract_data(master, condition='normal')
#     normal_scores = score_condition_by_person(L1_normal, L2_normal, people)

#     normal_dict = create_word_acc_std(
#         accuracy_scores_by_word=accuracy_over_people(normal_scores), 
#         standard_errors_by_word=std_error_over_people(normal_scores),
#         condition='normal'
#     )
#     master_normal[master] = normal_dict

# master_normal['human evaluators'] = condition_word_acc_dict['Normal']
# low_acc_words_dict, all_words = create_multiple_acc_plot_unaltered(master_normal, differentiate=False)

# # Create a figure for combining plots
# fig, axs = plt.subplots(2, 2, figsize=(28, 24), constrained_layout=False)
# axs = axs.flatten()

# for idx, (condition, condition_name) in enumerate(zip(conditions, condition_names)):
#     master_alterations = {}
#     for master in master_lists:
#         L1_condition, L2_condition = extract_data(master, condition=condition)
#         condition_scores = score_condition_by_person(L1_condition, L2_condition, people)

#         condition_dict = create_word_acc_std(
#             accuracy_scores_by_word=accuracy_over_people(condition_scores), 
#             standard_errors_by_word=std_error_over_people(condition_scores),
#             condition=condition
#         )

#         master_alterations[master] = condition_dict

#     master_alterations["human evaluators"] = condition_word_acc_dict[condition]
#     create_multiple_acc_plot_altered(axs[idx], master_alterations, low_acc_words_dict, all_words, condition_name)

# plt.subplots_adjust(hspace=0.4, wspace=0.4)
# plt.savefig('combined_plot.png', bbox_inches='tight')
# plt.show()









# ## Box and wisker plot ###
# data = {
#     ('Classifier A', 'Condition 1'): np.random.normal(80, 5, 100),
#     ('Classifier A', 'Condition 2'): np.random.normal(75, 5, 100),
#     ('Classifier B', 'Condition 1'): np.random.normal(85, 5, 100),
#     ('Classifier B', 'Condition 2'): np.random.normal(78, 5, 100),
#     # Add more data as needed
# }

# classifiers = ['Classifier A', 'Classifier B']
# conditions = ['Condition 1', 'Condition 2']
# colors = ['#1f77b4', '#ff7f0e']

# create_boxplot(data, classifiers, conditions, colors, title='Comparison of Classifiers Across Conditions', ylabel='Accuracy (%)')

# metrics = ['accuracy', 'precision', 'recall', 'fscore']
# for metric in metrics:
#     plot_metrics(metric=metric)





# dFirst = ['axe', 'own', 'all']
# dLast  = ['knee', 'tree', 'me', 'plow', 'few']

# condition_word_acc_dict = eval_human_for_big_plots()
