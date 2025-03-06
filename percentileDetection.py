import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans


def percentileDetection(df, selectedDataset):
    # Create a new dataframe with min and max values
    min_values = df.min()
    max_values = df.max()

    X = pd.DataFrame({'min': min_values, 'max': max_values})
    y = np.where(X.index.astype(int) == 0, 1, 0)

    if selectedDataset == "fmnist":
        malicious_id = 1
    elif selectedDataset == "fedemnist":
        malicious_id = 338
    elif selectedDataset == "cifar10":
        malicious_id = 4

    client_list = X.index.astype(int)
    malicious_list = np.array([int(c) for c in client_list if int(c) < malicious_id])

    plt.figure(figsize=(4,3))
    plt.scatter(x=X['min'], y=X['max'], c=y)
    # calculate the 75th percentile of the y-axis
    max_75_percentile = np.percentile(X['max'], 75)
    max_50_percentile = np.percentile(X['max'], 50)

    # calculate the 25th percentile of the x-axis
    min_25_percentile = np.percentile(X['min'], 25)
    min_50_percentile = np.percentile(X['min'], 50)

    # draw a vertical line at the 75th percentile of the y-axis
    plt.axhline(y=max_75_percentile, color='red', linestyle='--', label='75th percentile')
    plt.axhline(y=max_50_percentile, color='red', linestyle='--', label='Max 50th percentile')

    # draw a horizontal line at the 25th percentile of the x-axis
    plt.axvline(x=min_25_percentile, color='blue', linestyle='--', label='25th percentile')
    plt.axvline(x=min_50_percentile, color='blue', linestyle='--', label='Min 50th percentile')

    # show the legend
    plt.legend()
    plt.show()


   # benign_df = X[(X['max'] < max_75_percentile) &
   #                 (X['min'] > min_25_percentile)]
    benign_df = X[(X['max'] < max_50_percentile) &
                    (X['min'] > min_50_percentile)]
    malicious_df = X[~X.index.isin(benign_df.index.values)]

    #print (benign_df)
    print (f'predicted benign : {sorted(benign_df.index.values)}')
    print (f'predicted malicious : {sorted(malicious_df.index.values)}')
    return(benign_df.index.values, malicious_df.index.values, client_list, malicious_list)
