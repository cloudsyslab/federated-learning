import os, re, sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import glob
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, Birch
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import math
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score

from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cosine

def read_files(root_dir):
    file_list = glob.glob(root_dir + '/*.txt')
    df_list = []
    for file in file_list:
        col_name = os.path.basename(file).split('.')[0]
        df = pd.read_csv(file, header=None)
        df.columns = [col_name]
        df_list.append(df)
    return pd.concat(df_list, axis=1)


def formatit(client_list):
  return [c.split('_')[1] if '_' in c else c for c in client_list]

def evaluate(client_list, malicious_list, predicted_list):
  all_clients = formatit(client_list)
  y_true = [1 if all_clients[i] in malicious_list else 0 for i in range(len(all_clients))]
  y_pred = [1 if all_clients[i] in predicted_list else 0 for i in range(len(all_clients))]
  target_names = ['benign', 'malicious']
  #print (classification_report(y_true, y_pred, target_names=target_names))

  conf_matrix = confusion_matrix(y_true, y_pred)

  if len(conf_matrix) == 2:
    tn = conf_matrix[0, 0]  # True negatives
    fp = conf_matrix[0, 1]  # False positives
    fn = conf_matrix[1, 0]  # False negatives
    tp = conf_matrix[1, 1]  # True positives
  else:
    tn = conf_matrix.ravel()[0]
    fp = 0
    fn = 0
    tp = 0

  #print (f'Original  malicious to benign ratio: {len(malicious_list)/len(np.setdiff1d(all_clients,malicious_list)):.3f}')
  print (f'False Positives: {fp:<4}, False Negatives: {fn:<4}, True Negatives: {tn:<4}, malicious to benign ratio: {fn/tn:.3f}')
  # print (f'False Positive Rate: {fp/(fp+tn)} \nFalse Negative Rate: {fn/(tp+fn)}')
  # print (f'Balanced Accuracy: {balanced_accuracy_score(y_true, y_pred)}')

def call(algo, reduced_df, K, filter_clients=None, offset=0.5, topK=0.5):
  if filter_clients is not None:
    reduced_df = reduced_df.loc[filter_clients]

  #I don't think I need formatit cause my column names are just the client ID values
  client_list = np.array(formatit(reduced_df.index.values))
  #client_list = np.array(reduced_df.index.values)

  if algo == 'kmeans_clustering':
    clustering = KMeans(n_clusters=2).fit(reduced_df)
    indices = np.argwhere(clustering.labels_ == np.bincount(clustering.labels_).argmin()).flatten()
    #print (f'kmeans_clustering: {formatit(client_list[indices])}')
  elif algo == 'AgglomerativeClustering':
    clustering = AgglomerativeClustering().fit(reduced_df)
    indices = np.argwhere(clustering.labels_ == np.bincount(clustering.labels_).argmin()).flatten()
    print (f'AgglomerativeClustering: {formatit(client_list[indices])}')
  elif algo == 'Birch_clustering':
    clustering = Birch(n_clusters=2).fit(reduced_df)
    indices = np.argwhere(clustering.labels_ == np.bincount(clustering.labels_).argmin()).flatten()
    print (f'Birch_clustering: {formatit(client_list[indices])}')
  elif algo == 'Isolation_forest':
    clf = IsolationForest(random_state=1, n_estimators=100, max_features=max(2, math.floor(n/2)))
    predicted = clf.fit_predict(reduced_df)
    indices = np.argwhere(predicted == -1).flatten()
    #anomaly_scores = clf.decision_function(reduced_df)
    #scores = anomaly_scores - 0.5
    #print (f'Anomaly scores: {scores}')
    print (f'Isolation_forest: {formatit(client_list[indices])}')
  elif algo == 'Local_Outlier_Factor':
    clf = LocalOutlierFactor(n_neighbors=math.ceil(K/2))
    clf.fit_predict(reduced_df)
    lof = np.array(-1 * clf.negative_outlier_factor_)
    #print (f'outlier factor: {lof}')
    indices = np.argwhere(lof-offset > 1).flatten()
    #print (f'Lof predicted clients: {formatit(client_list[indices])}')
  elif algo == 'Local_Outlier_Factor_topK':
    clf = LocalOutlierFactor(n_neighbors=math.ceil(K/2))
    clf.fit_predict(reduced_df)
    lof = np.array(-1 * clf.negative_outlier_factor_)
    #print (f'outlier factor: {lof}')
    indices = np.argsort(lof)[:int(topK*K)]
    topKclients = client_list[indices]
    topKlof = lof[indices]
    indices = np.argwhere(topKlof-offset > 1).flatten()
    return topKclients[indices], topKclients, topKlof
  elif algo == 'DBSCAN clustering':
    neigh = NearestNeighbors(n_neighbors=K)
    nbrs = neigh.fit(reduced_df)
    distances, indices = nbrs.kneighbors(reduced_df)
    distances = distances[:,1:]
    dist_sorted = np.sort(distances.mean(axis=1))
    s_scoremax = 0
    for eps in dist_sorted:
      clustering = DBSCAN(eps=eps).fit(reduced_df)
      #print (clustering.labels_)
      if np.unique(clustering.labels_).size > 1:
        s_score = silhouette_score(reduced_df, clustering.labels_)
        #print (s_score)
        if s_score > s_scoremax:
          s_scoremax = s_score
          cluster_labels = clustering.labels_
    indices = np.argwhere(cluster_labels == -1).flatten()
    print (f'DBSCAN clustering: {formatit(client_list[indices])}')

  return client_list[indices]

def detect_malicious(selectedDataset, dataset, K, model, metrics):
  # Calculate minimum and maximum values for each column
  global_model = pd.read_csv("server_model.csv")["Server"].to_list()
  if(metrics == "minmax"):
    min_values = dataset.min()
    max_values = dataset.max()
    # Create a new dataframe with min and max values
    reduced_df = pd.DataFrame({'min': min_values, 'max': max_values})
  if(metrics == "EDCD"):
    print("running ECDC")
    reduced_df = pd.DataFrame(columns=["Euclidean Distances", "Cosine Distances"])
    for col in dataset:
      client_model = dataset[col].to_list()
      reduced_df.loc[len(reduced_df)] = euclidean(global_model, client_model), cosine(global_model, client_model)
    clientIDs = dataset.columns
    #print(clientIDs)
    reduced_df.set_index(clientIDs, inplace=True)

  #print(reduced_df)
  #print(dataset.columns)
  #print(reduced_df.index.values)
  

  client_list = np.array(formatit(reduced_df.index.values))
  reduced_df.index = client_list

  #client_list stores IDs as strings, so we cast them to ints so we can easily
  #sort them for printing
  intList = []
  intList = list(map(int, client_list))

  print (f'all client list: {sorted(intList)}')

  if selectedDataset == "fmnist":
    malicious_id = 1
  elif selectedDataset == "fedemnist":
    malicious_id = 338
  elif selectedDataset == "cifar10":
    malicious_id = 4

  malicious_list = np.array([c for c in client_list if int(c) < malicious_id])
  malicious_int_list = list(map(int, malicious_list))
  print (f'malicious client_list: {sorted(malicious_int_list)}')

  #Cluster visualization
  #print("Running visualization")
  #fig, ax = plt.subplots(figsize=(4,3))
  #color = ['red' if client in malicious_list else 'blue' for client in reduced_df.index]
  #reduced_df.plot.scatter(x='min',y='max',c=color,ax=ax)
  
  #  color = ['red' if client in malicious_list else 'blue' for client in reduced_df.index]
  #  fig, ax = plt.subplots(figsize=(4,3))
  #  reduced_df.plot.scatter(x='min',y='max',c=color,ax=ax)

  detection_metrics = {}

  if model == 'kmeans':
    predicted_malicious = call('kmeans_clustering', reduced_df, K)
    predicted_int_malicious = list(map(int, predicted_malicious))
    true_positives = []
    false_positives = []
    for value in predicted_int_malicious:
      if(value < malicious_id):
        true_positives.append(value)
      else:
        false_positives.append(value)

    predicted_benign = np.setdiff1d(client_list, predicted_malicious)
    predicted_int_benign = list(map(int, predicted_benign))
    true_negatives = []
    false_negatives = []
    for value in predicted_int_benign:
      if(value < malicious_id):
        false_negatives.append(value)
      else:
        true_negatives.append(value)
    
    print(f'true positives list:   {sorted(true_positives)}')
    print(f'false negatives list:  {sorted(false_negatives)}')
    print (f'Predicted malicious: {sorted(predicted_int_malicious)}')
    print (f'Predicted benign: {sorted(predicted_int_benign)}')

    detection_metrics = {
      "true_positives": sorted(true_positives),
      "false_negatives": sorted(false_negatives),
      "true_negatives": sorted(true_negatives),
      "false_positives": sorted(false_positives)
    }

    evaluate(client_list, malicious_list, predicted_malicious)
  elif model == 'lof':
    predicted_malicious = call('Local_Outlier_Factor', reduced_df, K, None, 0.1)
    predicted_int_malicious = list(map(int, predicted_malicious))
    true_positives = []
    false_positives = []
    for value in predicted_int_malicious:
      if(value < malicious_id):
        true_positives.append(value)
      else:
        false_positives.append(value)

    predicted_benign = np.setdiff1d(client_list, predicted_malicious)
    predicted_int_benign = list(map(int, predicted_benign))
    true_negatives = []
    false_negatives = []
    for value in predicted_int_benign:
      if(value < malicious_id):
        false_negatives.append(value)
      else:
        true_negatives.append(value)

    print(f'true positives list:   {sorted(true_positives)}')
    print(f'false negatives list:  {sorted(false_negatives)}')
    print (f'Predicted malicious:   {sorted(predicted_int_malicious)}')
    print (f'Predicted benign: {sorted(predicted_int_benign)}')

    detection_metrics = {
      "true_positives": sorted(true_positives),
      "false_negatives": sorted(false_negatives),
      "true_negatives": sorted(true_negatives),
      "false_positives": sorted(false_positives)
    }

    evaluate(client_list, malicious_list, predicted_malicious)
  elif model == 'lof_topk':
    predicted_malicious, topKclients, topKlof = call('Local_Outlier_Factor_topK', reduced_df, K, None, 0.1, 0.5)
    #predicted_malicious = np.concatenate((predicted_malicious, np.setdiff1d(malicious_list, topKclients)))
    predicted_benign = np.setdiff1d(topKclients, predicted_malicious)
    print (f'topkclients: {topKclients} {topKlof}')
    print (f'Predicted malicious: {predicted_malicious}')
    print (f'Predicted benign: {predicted_benign}')
    evaluate(topKclients, malicious_list, predicted_malicious)
  else:
    print ('invalid model name')
  
  all_selected_client_IDs = sorted(intList)
  return detection_metrics, all_selected_client_IDs, sorted(predicted_int_malicious)

  '''
  predicted_kmeans = call('kmeans_clustering', reduced_df, K, n)
  predicted_agglo = call('AgglomerativeClustering', reduced_df, K, n)
  predicted_birch = call('Birch_clustering', reduced_df, K, n)
  predicted_isolationf = call('Isolation_forest', reduced_df, K, n)
  predicted_dbscan = call('DBSCAN clustering', reduced_df, K, n)
  predicted_lof, topKclients, topKlof = call('Local_Outlier_Factor_topK', reduced_df, K, n, None, 0.1, 1.0)
  predicted_lof = np.concatenate((predicted_lof, np.setdiff1d(malicious_list, topKclients)))
  #print (f'Client ranking based on lof: {topKclients}')
  #print (f'Lof values in rank order: {[round(x,2) for x in topKlof]}')
  print (f'Predicted lof: {predicted_lof}')
  #print (f'Predicted benign: {np.setdiff1d(topKclients, predicted_lof)}')
  client_list = np.array(formatit(reduced_df.index.values))


  #K = len(predicted_lof)
  #if K > 2:
  #  predicted_lof = call('kmeans_clustering', reduced_df, K, n, predicted_lof)
  print (f'Predicted kmeans: {predicted_kmeans}')
  evaluate(client_list, malicious_list, predicted_kmeans)

  evaluate(client_list, malicious_list, predicted_lof)
  print (f'Predicted Agg clustering: {predicted_agglo}')
  evaluate(client_list, malicious_list, predicted_agglo)
  print (f'Predicted birch: {predicted_birch}')
  evaluate(client_list, malicious_list, predicted_birch)
  print (f'Predicted isolation forest: {predicted_isolationf}')
  evaluate(client_list, malicious_list, predicted_isolationf)
  '''

if __name__ == "__main__":
  # rounds = ['RND1_4', 'RND1_5', 'RND1_6','RND1_10','RND1_12']
  #rounds = ['RND1_1', 'RND1_2', 'RND1_4']
  if len(sys.argv) < 3:
    print("Usage: python script.py csv_file model")
    exit()

  table = sys.argv[1]
  model = sys.argv[2]
  #root_dir = "./detection_code/static_data/" + table
  #root_dir = "../data/Round1_only_67clients/" + table

  #for r in range(1,51):
  #for r in [5,6,45]:
  #  print ("\n" + root_dir + '/' + 'RND1_' + str(r))
  #  dataset = read_files(root_dir + '/' + 'RND1_' + str(r))
  #  print(dataset)
  #  K = len(dataset.columns)
  #  print ('---------------------RND1_'+str(r)+'--------------------')
  #  dataslice = dataset.tail(10).reset_index(drop=True)
  #  print(dataslice)
  #  detect_malicious(root_dir, dataslice, K, model)

  #dataslice = dataset.tail(10).reset_index(drop=True)
  dataslice = pd.read_csv(table)
  K = len(dataslice.columns)
  print(dataslice)
  detect_malicious("fedemnist", dataslice, K, model, "EDCD")
