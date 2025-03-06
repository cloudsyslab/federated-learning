import os, re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import glob

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score

def extract_features_minmax(dataset, selectedDataset):
  min_values = dataset.min()
  max_values = dataset.max()
  # Create a new dataframe with min and max values
  X = pd.DataFrame({'min': min_values, 'max': max_values})
  y = np.where(X.index.astype(int) < 338, 1, 0)
  #plt.figure(figsize=(4,3))
  #plt.scatter(x=X['min'], y=X['max'], c=y)
  #plt.show()

  if selectedDataset == "fmnist":
    malicious_id = 1
  elif selectedDataset == "fedemnist":
    malicious_id = 338
  elif selectedDataset == "cifar10":
    malicious_id = 4

  client_list = X.index.astype(int)
  malicious_list = np.array([int(c) for c in client_list if int(c) < malicious_id])

  return X.values, client_list, malicious_list

from sklearn.manifold import TSNE
def extract_features_tsne(dataset, selectedDataset):
  # Apply t-SNE to reduce dimensionality

  X = dataset.T
  y = np.where(X.index.astype(int) < 338, 1, 0)
  p = len(dataset.columns) - 1;
  divergence = []
  p_list = []
  while p >= 5:
    tsne = TSNE(n_components=2, random_state=42, perplexity=p)
    X_tsne = tsne.fit_transform(X)
    if len(divergence) >= 1:
      if tsne.kl_divergence_ > divergence[-1]:
        break
    divergence.append(tsne.kl_divergence_)
    p_list.append(p)
    p = p - 1

  # find the perplexity that gives minimum divergence
  p = p_list[divergence.index(min(divergence))]
  tsne = TSNE(n_components=2, random_state=42, perplexity=p)
  X_tsne = tsne.fit_transform(X)

  # Plot the results
  plt.figure(figsize=(4,3))
  plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y, cmap="Set1")
  plt.xlabel("t-SNE 1")
  plt.ylabel("t-SNE 2")
  plt.show()

  if selectedDataset == "fmnist":
    malicious_id = 1
  elif selectedDataset == "fedemnist":
    malicious_id = 338
  elif selectedDataset == "cifar10":
    malicious_id = 4

  client_list = X.index.astype(int)
  malicious_list = np.array([int(c) for c in client_list if int(c) < malicious_id])

  return X_tsne, client_list, malicious_list

from sklearn.cluster import KMeans
def kmeans_clustering(X, clients):
  n_clusters = 4
  model = KMeans(n_clusters, random_state=42)
  model.fit(X) # Fit the model to the data
  # Get the cluster labels and centroids
  labels = model.labels_ # An array of cluster labels for each point
  centroids = model.cluster_centers_ # An array of cluster centroids
  client_list = []
  wcss = [] #Within-Cluster Sum of Squares for each cluster
  cluster_size = []

  for k in range(n_clusters):
    cluster_k = X[labels==k]
    clients_k = clients[labels==k]
    center_k = centroids[k]
    sq_distances = np.sum((cluster_k - center_k) ** 2, axis=1)
    wcss_k = np.sum(sq_distances)
    client_list.append(sorted(list(map(int, clients_k))))
    wcss.append(wcss_k)
    cluster_size.append(len(cluster_k))

  wcss_df = pd.DataFrame(list(zip(client_list, centroids, wcss, cluster_size)), columns=["clients","centroid","wcss","size"])

  # First, find the index of the row that has the largest value in size (consider it as benign)
  max_size_idx = wcss_df['size'].idxmax()
  max_size = wcss_df.loc[max_size_idx, 'size']
  # if there is a tie, find the index of the row that has the lowest value in wcss among the rows that have the max size
  min_wcss_idx = wcss_df[wcss_df['size'] == wcss_df.loc[max_size_idx, 'size']]['wcss'].idxmin()
  # Third, drop that row from the dataframe
  wcss_df = wcss_df.drop(min_wcss_idx)

  # consider all remaining clusters as malicious.
  predicted = []
  for i in wcss_df.index.tolist():
    predicted = predicted + client_list[i]

  # Plot the clustering results
  plt.figure(figsize=(4,3))
  plt.scatter(X[:,0], X[:,1], c=labels, cmap='Set1') # Plot the points with different colors for each cluster
  plt.scatter(centroids[:,0], centroids[:,1], marker='*', s=200, c='black')
  return (predicted)

from sklearn.neighbors import LocalOutlierFactor
def local_outlier_factor(X, clients, offset):
  K = len(clients)
  model = LocalOutlierFactor(n_neighbors=int(K/2))
  #model = LocalOutlierFactor(n_neighbors=int(K/2), metric='cosine')
  outlier_prediction = model.fit_predict(X)
  lof = np.array(-1 * model.negative_outlier_factor_)
  topindices = np.argsort(lof)[:int(0.5*K)] # top 50% of clients with low LOF values (likely to be benign)
  topclients = clients[topindices].values
  bottomindices = np.argsort(lof)[int(0.5*K):] # bottom 50% clients with high LOF value (discard as malicious)
  bottomclients = clients[bottomindices].values
  top_loflist = lof[topindices]
  benign_indices = np.argwhere(top_loflist-offset <= 1).flatten() # considered benign among top 50% clients
  predicted_benign = topclients[benign_indices]
  malicious_indices = np.argwhere(top_loflist-offset > 1).flatten() # considered malicious among top 50% clients
  print (top_loflist, topclients, topclients[malicious_indices])
  if len(malicious_indices) >= 1:
    predicted_malicious = np.concatenate((topclients[malicious_indices], bottomclients), axis=0)
  else:
    predicted_malicious = bottomclients
  return predicted_benign, predicted_malicious

#def local_outlier_factor(X, clients, offset):
#  K = len(clients)
#  model = LocalOutlierFactor(n_neighbors=int(K/2))
#  outlier_prediction = model.fit_predict(X)
#  lof = np.array(-1 * model.negative_outlier_factor_)
  #print ('sorted lof:', lof[np.argsort(lof)])
  #print ('sorted clients:', clients[np.argsort(lof)].values)
#  topindices = np.argsort(lof)[:int(0.5*K)] # top 50% of clients with low LOF values (likely to be benign)
#  bottomindices = np.argsort(lof)[int(0.5*K):] # bottom 50% clients with high LOF value (discard as malicious)
#  bottomclients = clients[bottomindices].values
#  top_loflist = lof[topindices]
#  benign_indices = np.argwhere(top_loflist-offset <= 1).flatten() # considered benign among top 50% clients
#  predicted_benign = clients[benign_indices].values
#  malicious_indices = np.argwhere(top_loflist-offset > 1).flatten() # considered malicious among top 50% clients

#  if len(malicious_indices) >= 1:
#    predicted_malicious = np.concatenate((clients[malicious_indices].values, bottomclients), axis=0)
#  else:
#    predicted_malicious = bottomclients
#  return predicted_benign, predicted_malicious

def evaluate(client_list, malicious_list, predicted_list, f):
  y_true = [1 if client in malicious_list else 0 for client in client_list]
  y_pred = [1 if client in predicted_list else 0 for client in client_list]

  target_names = ['benign', 'malicious']
  #print (classification_report(y_true, y_pred, target_names=target_names))
  tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
  print (f'{roundinfo} False Positives: {fp:<4}, False Negatives: {fn:<4}, True Positives: {tp:<4}', file=f)
