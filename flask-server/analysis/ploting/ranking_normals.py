from data.constants import word_list
from analysis.json_helpers import loadDictFromJSON
from analysis.analysis_functions import loadXFromSavedJson

import os
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from sklearn.metrics import r2_score 

import numpy as np

def makeCoolPlot(colors: list, data: dict, list_number: int, grid_type):
    # Define the colors and color limits
    cmap = ListedColormap(colors)
    vmin = 0
    vmax = 2

    # Generate some example data
    plot_data = []
    plot_y_labels = []

    for instance in data:
        plot_data.append(instance['data'])
        plot_y_labels.append(instance['name'])

    # Create the pseudocolor plot
    plt.imshow(plot_data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

    cbar = plt.colorbar()
    cbar.set_ticks([0.33, 1, 1.66])
    cbar.set_ticklabels(['No Response', 'Incorrect Response', 'Correct Response'])
    for tick in cbar.ax.get_yticklabels():
        tick.set_weight('bold')

    # Customize the ticks and labels
    plt.xticks(np.arange(0, 50, 1), word_list[list_number], rotation=90)
    plt.yticks(np.arange(0, len(plot_y_labels), 1), plot_y_labels)
    plt.tick_params(axis='both', length=0)

    # Add minor ticks and gridlines
    minor_locator = MultipleLocator(0.5)
    plt.gca().xaxis.set_minor_locator(minor_locator)
    plt.gca().yaxis.set_minor_locator(minor_locator)
    plt.grid(which='minor', axis='both', color='gray', linestyle='-', linewidth=0.5)

    # Set the limits of the x-axis and y-axis to extend the gridlines to the edges
    plt.xlim([-0.5, 49.5])
    plt.ylim([-0.5, len(plot_y_labels) - 0.5])

    # Set the title and axis labels
    plt.title('API Responses, List {}, {}'.format(list_number, grid_type), fontweight='bold')
    plt.xlabel('Word Tested',fontweight='bold')
    plt.ylabel('Word List', fontweight='bold')

    # Display the plot
    plt.show()

def makeCoolPlotDetermination(data: dict, list_number: int, grid_type):
    # Define the colors and color limits
    colors = ['#ff0000', '#00ff00']
    cmap = ListedColormap(colors)
    vmin = 0
    vmax = 1

    # Generate some example data
    plot_data = []
    plot_y_labels = []

    for instance in data:
        plot_data.append(instance['data'])
        plot_y_labels.append(instance['name'])

    # Create the pseudocolor plot
    plt.imshow(plot_data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

    cbar = plt.colorbar(ticks=[0, 1])
    cbar.set_ticklabels(['Pronounced Incorrect','Pronounced Correct'])
    for tick in cbar.ax.get_yticklabels():
        tick.set_weight('bold')

    # Customize the ticks and labels
    plt.xticks(np.arange(0, 50, 1), word_list[list_number], rotation=90)
    plt.yticks(np.arange(0, len(plot_y_labels), 1), plot_y_labels)
    plt.tick_params(axis='both', length=0)

    # Add minor ticks and gridlines
    minor_locator = MultipleLocator(0.5)
    plt.gca().xaxis.set_minor_locator(minor_locator)
    plt.gca().yaxis.set_minor_locator(minor_locator)
    plt.grid(which='minor', axis='both', color='gray', linestyle='-', linewidth=0.5)

    # Set the limits of the x-axis and y-axis to extend the gridlines to the edges
    plt.xlim([-0.5, 49.5])
    plt.ylim([-0.5, len(plot_y_labels) - 0.5])

    # Set the title and axis labels
    plt.title('API Responses, List {}, {}'.format(list_number, grid_type), fontweight='bold')
    plt.xlabel('Word Tested',fontweight='bold')
    plt.ylabel('Word List', fontweight='bold')

    # Display the plot
    plt.show()

def makeCoolPlotAccuracy(data: dict, list_number: int, grid_type):
    # Define the colors and color limits
    color_map = plt.cm.get_cmap('plasma', 10)
    norm = Normalize(vmin=0, vmax=1)

    # Generate some example data
    plot_data = []
    plot_y_labels = []

    for instance in data:
        plot_data.append(instance['data'])
        plot_y_labels.append(instance['name'])

    # Create the pseudocolor plot
    plt.imshow(plot_data, cmap=color_map, norm=norm, aspect='auto')

    # Add the colorbar
    sm = ScalarMappable(cmap=color_map, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ticks=np.linspace(1, 0, 10))
    cbar.ax.set_yticklabels(['1.0', '0.9', '0.8', '0.7', '0.6', '0.5', '0.4', '0.3', '0.2', '0.1'])
    for tick in cbar.ax.get_yticklabels():
        tick.set_weight('bold')

    # Customize the ticks and labels
    plt.xticks(np.arange(0, 50, 1), word_list[list_number], rotation=90)
    plt.yticks(np.arange(0, len(plot_y_labels), 1), plot_y_labels)
    plt.tick_params(axis='both', length=0)

    # Add minor ticks and gridlines
    minor_locator = MultipleLocator(0.5)
    plt.gca().xaxis.set_minor_locator(minor_locator)
    plt.gca().yaxis.set_minor_locator(minor_locator)
    plt.grid(which='minor', axis='both', color='gray', linestyle='-', linewidth=0.5)

    # Set the limits of the x-axis and y-axis to extend the gridlines to the edges
    plt.xlim([-0.5, 49.5])
    plt.ylim([-0.5, len(plot_y_labels) - 0.5])

    # Set the title and axis labels
    plt.title('API Responses, List {}, {}'.format(list_number, grid_type), fontweight='bold')
    plt.xlabel('Word Tested',fontweight='bold')
    plt.ylabel('Word List', fontweight='bold')

    # Display the plot
    plt.show()

def confadinceScatterPlot(data: dict, list_number: int, name):
    # Extract x and y values from the data_dict
    words = word_list[list_number]
    domain = np.linspace(1, 50, 50)

    long_list = np.array([])
    for normal in data:
        x = normal['data']
        y = domain
        plt.scatter(y, x, label=normal['name'])
        long_list = np.append(long_list, x, axis=0)

    mean = np.mean(long_list)

    for i in range(1, len(words)+1):
        plt.vlines(i, 0, 1, colors='gray', linestyles='dashed', alpha=0.5)

    plt.axhline(y=mean, color='red', linestyle='--', label='Mean')
    
    plt.ylim(0,1)
    plt.legend(loc='lower right')
    plt.xticks(domain, words, rotation=90)
    plt.title('Classification Confadince: {}'.format(name), fontweight='bold')
    plt.xlabel('Word Tested, List {}'.format(list_number),fontweight='bold')
    plt.ylabel('Confadince', fontweight='bold')
    plt.show()

def violinPlot(list_number: int, json_path: str):
    data = loadXFromSavedJson('confidence', list_number, json_path)
    name = json_path.split('/')[-2]
    words = word_list[list_number]
    domain = np.linspace(1, 50, 50)

    # organizing the data via word not via trial
    twoDim_data = [
        [],[],[],[],[],[],[],[],[],[],
        [],[],[],[],[],[],[],[],[],[],
        [],[],[],[],[],[],[],[],[],[],
        [],[],[],[],[],[],[],[],[],[],
        [],[],[],[],[],[],[],[],[],[],
    ]
    for x in range(len(words)):
        for normal in data:
            sub = normal['data']
            twoDim_data[x].append(sub[x])

    # sorting twoDim_Data by lowest mean confadince
    means = []
    for subArray in twoDim_data:
        # calculating the mean of each sub array
        means.append(sum(subArray) / len(subArray))

    # list of tuples, each containing the index of a subarray and its mean value
    index_means = list(enumerate(means))

    # Sort the list of tuples based on the mean value
    sorted_index_means = sorted(index_means, key=lambda x: x[1])

    # sorted the 2d data and assoiated words
    sorted_twoDim_data = [twoDim_data[index] for index, mean in sorted_index_means]
    sorted_words = [words[index] for index, mean in sorted_index_means]

    # Create a figure and axes object
    fig, ax = plt.subplots()

    # Create the violin plot
    parts = ax.violinplot(sorted_twoDim_data, showmeans=True, showmedians=False, showextrema=True)

    for i in range(1, len(words)+1):
        plt.vlines(i, 0, 1, colors='gray', linestyles='dashed', alpha=0.5)

    plt.ylim(0,1)
    plt.xticks(domain, sorted_words, rotation=90)
    plt.title('Classification Confadince: {}'.format(name), fontweight='bold')
    plt.xlabel('Word Tested, List {}'.format(list_number),fontweight='bold')
    plt.ylabel('Confadince', fontweight='bold')
    plt.show()

def accuracyBarPlot(list_number: int, json_path: str):
    accData = loadXFromSavedJson('accuracy', list_number, json_path)
    conData = loadXFromSavedJson('confidence', list_number, json_path)
    name = json_path.split('/')[-2]
    words = word_list[list_number]

    # organizing the data via word not via trial
    twoDim_con_data = [
        [],[],[],[],[],[],[],[],[],[],
        [],[],[],[],[],[],[],[],[],[],
        [],[],[],[],[],[],[],[],[],[],
        [],[],[],[],[],[],[],[],[],[],
        [],[],[],[],[],[],[],[],[],[],
    ]
    for x in range(len(words)):
        for normal in conData:
            sub = normal['data']
            twoDim_con_data[x].append(sub[x])

    # sorting twoDim_con_data by lowest mean confadince
    means = []
    for subArray in twoDim_con_data:
        # calculating the mean of each sub array
        means.append(sum(subArray) / len(subArray))

    # list of tuples, each containing the index of a subarray and its mean value
    index_means = list(enumerate(means))

    # Sort the list of tuples based on the mean value
    sorted_index_means = sorted(index_means, key=lambda x: x[1])

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
    
    # sorted the 2d data and assoiated words
    sorted_accuracy = [accuracy[index] for index, mean in sorted_index_means]
    sorted_words = [words[index] for index, mean in sorted_index_means]

    plt.bar(sorted_words, sorted_accuracy)

    plt.ylim(0,1)
    plt.xticks(rotation=90)
    # plt.title('Classification Accuracy: {}'.format(name), fontweight='bold')
    plt.title('Classification Accuracy: L2_dLast', fontweight='bold')
    plt.xlabel('Word Tested, List {}'.format(list_number),fontweight='bold')
    plt.ylabel('Accuracy', fontweight='bold')
    plt.show()

def accVsCon(list_number: int, json_path: str):
    words = word_list[list_number]
    name = json_path.split('/')[-2]
    accData = loadXFromSavedJson('accuracy', list_number, json_path)
    conData = loadXFromSavedJson('confidence', list_number, json_path)

    # organizing the data via word not via trial
    twoDim_con_data = [
        [],[],[],[],[],[],[],[],[],[],
        [],[],[],[],[],[],[],[],[],[],
        [],[],[],[],[],[],[],[],[],[],
        [],[],[],[],[],[],[],[],[],[],
        [],[],[],[],[],[],[],[],[],[],
    ]
    
    for x in range(len(words)):
        for normal in conData:
            sub = normal['data']
            twoDim_con_data[x].append(sub[x])

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

    confidance = []
    for word in twoDim_con_data:
        val = sum(word) / len(word)
        confidance.append(round(val,2))

    accuracy = []
    for word in twoDim_acc_data:
        val = sum(word) / len(word)
        accuracy.append(round(val,2))

    print(accuracy)
    print(confidance)

    type = 'plot'

    if type == 'plot':
        coefficients = np.polyfit(confidance, accuracy, 1)
        m, b = coefficients
        p = np.poly1d(coefficients)

        y_pred = p(confidance)
        r2 = r2_score(accuracy, y_pred)

        plt.scatter(confidance, accuracy)

        plt.plot(np.linspace(0, 1), p(np.linspace(0, 1)), 'r')

        plt.ylim(-0.1,1.1)
        plt.xlim(-0.1,1.1)
        # plt.xticks(rotation=90)
        plt.title('Per Word Acc vs Con, (Trials Ave): {}'.format(name), fontweight='bold')
        plt.ylabel('Accuracy',fontweight='bold')
        plt.xlabel('Confidence', fontweight='bold')
        #plt.text(0.05, 0.8, f"R^2 = {r2:.2f}", fontsize=12, bbox=dict(facecolor='red', alpha=0.5), transform=plt.gca().transAxes)
        plt.text(0.05, 0.9, f"y={coefficients[0]:.2f}x+{coefficients[1]:.2f}", bbox=dict(facecolor='red', alpha=0.5), fontsize=12, transform=plt.gca().transAxes)
        plt.show()

    elif type == 'table':
        data = np.array([word_list[list_number], accuracy, confidance]).T.tolist()
        # data = [word_list[list_number],accuracy,confidance]
        fig, ax = plt.subplots(figsize=(6,4))

        table = ax.table(cellText=data, loc='center')
        table.set_fontsize(14)
        table.scale(1.2, 1.2)  # Adjust the table size if needed

        # # Center align the cell values and make them bold
        # for i, cell in enumerate(table._cells):
        #     if i % len(column1) == 0:  # Skip the header row
        #         continue
        #     cell.set_text_props(weight='bold', ha='center')

        plt.subplots_adjust(left=0.2, bottom=0.2)
        ax.axis('off')
        plt.show()

def two_dimensional_grid(grid_type: str, list_number: int, json_path: str):

    if grid_type == 'deletion':

        for folder in folders:
            path = json_path + folder['name'] +'/'

            data = []
            for index in range(50):
                file_name = '{}_{}'.format(index,folder['name'].rsplit('_', 1)[0] + '.json')
                loaded_json = loadDictFromJSON(file_name, path)
                loaded_json = loaded_json['results']

                if len(loaded_json) == 0: 
                    data.append(0) # 0 => no API Responce
                elif len(loaded_json) > 0:
                    content = loaded_json[0]['alternatives'][0]['content'].lower()

                    # # Debugging 
                    # print()
                    # print('Index: {}'.format(index))
                    # print('Current: {}'.format(content))
                    # print('Word List: {}'.format(word_list[list_number][index]))

                    if content == word_list[list_number][index]:
                        data.append(2) # 2 => correct API Responce
                        print(2)
                    else:
                        data.append(1) # 1 => incorrect API responce
                        print(1)

            folders_with_data.extend([{'name': folder['name'], 'data': data}])
            colors = ['#00ff00', '#ffff00', '#ff0000']
        makeCoolPlot(colors, folders_with_data, list_number, grid_type)

    elif grid_type == 'normal':

        for folder in folders:
            path = json_path + folder['name'] +'/'

            data = []
            for index in range(50):
                file_name = '{}_{}'.format(index,folder['name'] + '.json')
                loaded_json = loadDictFromJSON(file_name, path)
                loaded_json = loaded_json['results']

                if len(loaded_json) == 0: 
                    data.append(0) # 0 => no API Responce
                elif len(loaded_json) > 0:
                    content = loaded_json[0]['alternatives'][0]['content'].lower()

                    # # Debugging
                    # print()
                    # print('Index: {}'.format(index))
                    # print('Current: {}'.format(content))
                    # print('Word List: {}'.format(word_list[list_number][index]))

                    if content == word_list[list_number][index]:
                        data.append(2) # 2 => correct API Responce
                        print(2)
                    else:
                        data.append(1) # 1 => incorrect API responce
                        print(1)

            folders_with_data.extend([{'name': folder['name'], 'data': data}])
            colors = ['#ff0000', '#ffff00', '#00ff00']
        makeCoolPlot(colors, folders_with_data, list_number, grid_type)

    elif grid_type == 'accuracy':

        for folder in folders:
            path = json_path + folder['name'] +'/'

            data = []
            for index in range(50):
                #file_name = '{}_{}'.format(index,folder['name'] + '.json')
                file_name = '{}_{}'.format(index,folder['name'].rsplit('_', 1)[0] + '.json')

                loaded_json = loadDictFromJSON(file_name, path)
                loaded_json = loaded_json['results']

                if len(loaded_json) == 0: 
                    data.append(0) # 0 => no API Responce
                elif len(loaded_json) > 0:
                    accuracy = loaded_json[0]['alternatives'][0]['confidence']
                    data.append(accuracy)

            folders_with_data.extend([{'name': folder['name'], 'data': data}])
        makeCoolPlotAccuracy(folders_with_data, list_number, grid_type)

    elif grid_type == 'determination':

        for folder in folders:
            path = json_path + folder['name'] +'/'

            data = []
            for index in range(50):
                file_name = '{}_{}'.format(index,folder['name'] + '.json')
                #file_name = '{}_{}'.format(index,folder['name'].rsplit('_', 1)[0] + '.json')

                loaded_json = loadDictFromJSON(file_name, path)
                loaded_json = loaded_json['results']

                if len(loaded_json) > 0:
                    accuracy = loaded_json[0]['alternatives'][0]['confidence']
                    content = loaded_json[0]['alternatives'][0]['content'].lower()
                    ref_word = word_list[list_number][index]

                    if content == ref_word and accuracy > 0.80: 
                        data.append(1)
                        
                        # Debugging
                        # print()
                        # print('Correct Identified')
                        # print('Index: {}'.format(index))
                        # print('Accuracy: {}'.format(accuracy))
                        # print('Content: {}'.format(content))
                        # print('Ref_word: {}'.format(ref_word))
                    else:
                        data.append(0)
                else:
                    data.append(0)

            folders_with_data.extend([{'name': folder['name'], 'data': data}])
            print(folders_with_data)
        makeCoolPlotDetermination(folders_with_data, list_number, grid_type)

    elif grid_type == 'scatter':

        for folder in folders:
            path = json_path + folder['name'] +'/'

            data = []
            for index in range(50):

                if name[3:] == 'normal':
                    file_name = '{}_{}'.format(index,folder['name'] + '.json')
                elif name[3:] == 'deletion':
                    file_name = '{}_{}'.format(index,folder['name'].rsplit('_', 1)[0] + '.json')

                loaded_json = loadDictFromJSON(file_name, path)
                loaded_json = loaded_json['results']

                if len(loaded_json) > 0:
                    confidence = loaded_json[0]['alternatives'][0]['confidence']
                    content = loaded_json[0]['alternatives'][0]['content'].lower()
                    ref_word = word_list[list_number][index]
                    
                    data.append(confidence)
                else:
                    data.append(0)

            folders_with_data.extend([{'name': folder['name'], 'data': data}])

        confadinceScatterPlot(folders_with_data, list_number, name)