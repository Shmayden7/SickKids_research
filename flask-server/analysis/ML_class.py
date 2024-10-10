import sklearn.datasets
import matplotlib.pyplot as plt
import numpy as np


def dumb_python():
    data_x, data_y = sklearn.datasets.make_blobs(centers=[[-2, -2], [2, 2]], 
                                             cluster_std=[0.3, 1.5], 
                                             random_state=0, 
                                             n_samples=200, 
                                             n_features=2)

    print(f'Number of points: {len(data_x)}')

    red_points = []
    blue_points = []
    for x, y in zip(data_x, data_y):
        # Red Condition
        if y == 0:
            red_points.append(x)
        # Blue Condition
        elif y == 1:
            blue_points.append(x)
            
    # Sanity check
    print(f'Number of Red points: {len(red_points)}')
    print()
    print(f'Number of Blue points: {len(blue_points)}')
    print()
    print(f'Sum: {len(red_points)+len(blue_points)}')

    plt.figure(figsize=(6,6))
    for red_vals, blue_vals in zip(red_points, blue_points):
        plt.scatter(red_vals[0], red_vals[1], label='Initial y=0',color='red')
        plt.scatter(blue_vals[0], blue_vals[1], label='Initial y=1',color='blue')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.show()