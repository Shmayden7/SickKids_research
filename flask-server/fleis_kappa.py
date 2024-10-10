from analysis.accuracy_plots import get_fleis_kappa, get_combined_fleiss_kappa, get_percent_agreement, get_percent_agreement_specific, get_combined_fleiss_kappa_specific

classifiers = ['ursa.json', 'whisper large.json', 'whisper medium.json']
human_raters = ['cole', 'joyce', 'ryley', 'lulia', 'isabel', 'susan', 'maria']
raters_to_evaluate = ['ursa.json', 'whisper large.json', 'whisper medium.json', 'cole', 'joyce', 'ryley', 'lulia', 'isabel', 'susan', 'maria']
conditions = ['normal', 'dFirst', 'dLast', 'c1000', 'c2000', 'c4000', 'c6000']

print("Only Machines")
print(get_combined_fleiss_kappa(classifiers, human_raters, classifiers))
print("Only Humans")
print(get_combined_fleiss_kappa(classifiers, human_raters, human_raters))
print("Machines and Humans")
print(get_combined_fleiss_kappa(classifiers, human_raters, raters_to_evaluate))

print()

# for condition in conditions:
#     temp = [condition]
#     print(f'Condition: {condition}')
#     print(f"Only Machines K: {get_combined_fleiss_kappa_specific(classifiers, human_raters, classifiers, temp)}")
#     print(f"Only Humans K: {get_combined_fleiss_kappa_specific(classifiers, human_raters, human_raters, temp)}")
#     print(f"Machines and Humans K: {get_combined_fleiss_kappa_specific(classifiers, human_raters, raters_to_evaluate, temp)}")
#     print()

# for machine in classifiers:
#     temp = [machine]
#     print(f'Machine: {machine}')
#     print(f"K: {get_combined_fleiss_kappa(temp, human_raters, temp)}")
#     print()

# print(get_combined_fleiss_kappa_specific(classifiers, human_raters, classifiers, ['dFirst', 'dLast', 'c1000', 'c2000', 'c4000', 'c6000']))
# print(get_combined_fleiss_kappa_specific(classifiers, human_raters, human_raters, ['dFirst', 'dLast', 'c1000', 'c2000', 'c4000', 'c6000']))
# print(get_combined_fleiss_kappa_specific(classifiers, human_raters, raters_to_evaluate, ['dFirst', 'dLast', 'c1000', 'c2000', 'c4000', 'c6000']))
