from typing import Dict, List, Tuple, Optional, Union
from functools import reduce
from collections import OrderedDict
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from flwr.common import Metrics, Scalar, EvaluateRes, FitRes, parameters_to_ndarrays, ndarrays_to_parameters, NDArray, NDArrays
from flwr.server.client_proxy import ClientProxy
from utils import H5Dataset
import datetime

import numpy as np
import copy

import flwr as fl
import torch
from pathlib import Path
import utils
import math
import warnings

import our_detect_model_poisoning
import our_detection_v2
import our_detection_v3
import percentileDetection

warnings.filterwarnings("ignore")

def get_on_fit_config_fn():

    def fit_config(server_round: int):
        """Return training configuration dict for each round.

        Keep batch size fixed at 32, perform two rounds of training with one
        local epoch, increase to two local epochs afterwards.
        """
        new_list = ""
        if selectedDataset == "fedemnist":
            #id_list = np.random.choice(3383, math.floor(3383*.01), replace=False)
            id_list = np.random.choice(3383, 66, replace=False)
            #print("ID list:")
            #print(id_list)
            #The ID list has to be converted to a string because Flower won't except a list as a config option
            new_list = ""
            for item in id_list:
                new_list += " " + str(item)
        #new_list is used by the clients to select a dataslice during fedemnist test...it is not needed for cifar 
        else:
            new_list == ""
        #print(new_list)
        config = {
            "batch_size": 64 if selectedDataset == "fedemnist" else 256,
            "current_round": server_round,
            "local_epochs": 10 if selectedDataset == "fedemnist" else 2,
            "id_list": new_list,
        }
        return config
    
    return fit_config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 1 #if server_round < 4 else 10
    return {"val_steps": val_steps}


def get_evaluate_fn(model: torch.nn.Module, toy: bool, data):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    trainset, testset, _ = utils.load_data(data)

    n_train = len(trainset)
    if toy:
        # use only 10 samples as validation set
        valset = torch.utils.data.Subset(trainset, range(n_train - 10, n_train))
        idxs = (testset.targets == 5).nonzero().flatten().tolist()
        poisoned_valset = utils.DatasetSplit(copy.deepcopy(testset), idxs)
        utils.poison_dataset(poisoned_valset.dataset, "cifar10", idxs, poison_all=True)
    else:
        # Use the last 5k training examples as a validation set
        #valset = torch.utils.data.Subset(trainset, range(n_train - 5000, n_train))
        if(selectedDataset == "cifar10"):
            idxs = (testset.targets == 5).nonzero().flatten().tolist()
            poisoned_valset = utils.DatasetSplit(copy.deepcopy(testset), idxs)
            utils.poison_dataset(poisoned_valset.dataset, "cifar10", idxs, poison_all=True)
        elif(selectedDataset == "fmnist"):
            idxs = (testset.targets == 5).nonzero().flatten().tolist()
            poisoned_valset = utils.DatasetSplit(copy.deepcopy(testset), idxs)
            utils.poison_dataset(poisoned_valset.dataset, "fmnist", idxs, poison_all=True)
        elif(selectedDataset == "fedemnist"):
            #poisoned_valset = copy.deepcopy(testset)
            idxs = (testset.targets == 5).nonzero().flatten().tolist()
            poisoned_valset = utils.DatasetSplit(copy.deepcopy(testset), idxs)
            utils.poison_dataset(poisoned_valset.dataset, selectedDataset, idxs, poison_all=True)

    if(selectedDataset == "cifar10" or selectedDataset == "fmnist"):
        valLoader = DataLoader(testset, batch_size=256, shuffle=False)
        poisoned_val_loader = DataLoader(poisoned_valset, 256, shuffle=False)
    else:
        valLoader = DataLoader(testset, batch_size=64, shuffle=False)
        poisoned_val_loader = DataLoader(poisoned_valset, 64, shuffle=False)

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        

        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        model.to(device)
        # Update model with the latest parameters
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        #params_dict = zip(model.state_dict().keys(), parameters)
        #state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        #model.load_state_dict(state_dict, strict=True)


        loss, accuracy, per_class_accuracy = utils.test(model, valLoader)
        poison_loss, poison_accuracy, poison_per_class_accuracy = utils.test(model, poisoned_val_loader)
        
        if(selectedDataset == "cifar10"):
            with open(filename, "a") as f:
                print("Round {} accuracy: {}".format(str(server_round), accuracy), file=f)
                print("Round {} poison accuracy: {}\n".format(str(server_round), poison_accuracy), file=f)
        else:
            with open(filename, "a") as f:
                print("Round {} accuracy: {}".format(str(server_round), accuracy), file=f)
                print("Round {} poison accuracy: {}\n".format(str(server_round), poison_accuracy), file=f)

        if(loss == "nan"):
            print("LOSS IS NAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAN")
        return loss, {"accuracy": accuracy, "per_class_accuracy": per_class_accuracy, "poison_accuracy": poison_accuracy}

    return evaluate

class AggregateCustomMetricStrategy(fl.server.strategy.FedAvgM):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}
        
        #for client in results:
        #   print("Client: " + str(client[1].metrics))
        #    for metric in client[1].metrics:
        #        print(metric)

        # Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if(selectedDataset == "cifar10"):
            with open(filename, "a") as f:
                print("Server Round {}".format(str(server_round)), file=f)
        else:
            with open(filename, "a") as f:
                print("Server Round {}".format(str(server_round)), file=f)

        #new custom aggregation (delta value implementation)

        #number of cifar model parameters
        if selectedDataset == "cifar10":
            n_params = 537610
        #number of fedemnist model parameters 
        else:
            n_params = 1199882

        #lr_vector = torch.Tensor([self.server_learning_rate]*n_params)
        lr_vector = np.array([self.server_learning_rate]*n_params)

        update_dict = {}
        #Construct the UTD update dict for detection purposes
        for _, r in results:
            #print("THE CLIENTS ID IS: ")
            #print(r.metrics["clientID"])
            #print(r.parameters)

            #select the correct model for the selected dataset
            if selectedDataset == "cifar10":
                model = utils.Net()
            else:
                model = utils.CNN_MNIST()

            #doing this sequence of commands allows us to convert the params from the form flower uses to the form UTD uses
            params_dict = zip(model.state_dict().keys(), parameters_to_ndarrays(r.parameters))
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=False)
            UTD_test = parameters_to_vector(model.parameters()).detach()
            update_dict[r.metrics["clientID"]] = UTD_test

        #Checking if the update dict is being created correctly 
        #print("UPDATE DICT")
        #print(update_dict)
            
        #convert data into a dataframe for our detection code
        if(ourDetect):
            if(selectedDataset == "cifar10"):
                with open(filename, "a") as f:
                    print("RUNNING OUR DETECTION", file=f)
                    print("Using {} clustering".format(cluster_algorithm), file=f)
            else:
                with open(filename, "a") as f:
                    print("RUNNING OUR DETECTION", file=f)
                    print("Using {} clustering".format(cluster_algorithm), file=f)

            print("RUNNING OUR DETECTION")
            df = pd.DataFrame(update_dict)
            #print(df)
            K = len(df.columns)
            #full_model = df.to_csv('Round1_cifar_full_client_models.csv', index=False)
            detection_slice = df.tail(10).reset_index(drop=True)
            for column in detection_slice.columns:
                #print(column)
                detection_slice.rename({column: "Client_" + str(column)}, axis=1, inplace=True)
            #print(detection_slice)
            #save the df to a csv for testing
            #saved_csv = detection_slice.to_csv('10_rounds_tests/testsForPaperGraphs/fmnist/fmnist_Test1_Round{}_client_models.csv'.format(str(server_round)), index=False)
            #call our detection code
            detection_metrics, all_clients, predicted_malicious = our_detect_model_poisoning.detect_malicious(selectedDataset, detection_slice, K, cluster_algorithm, "minmax")
            print("The predicted malicious clients")
            print(sorted(predicted_malicious))

            if(selectedDataset == "cifar10" or selectedDataset == "fmnist"):
                with open(filename, "a") as f:
                    print("The predicted malicious clients: {}".format(predicted_malicious), file=f)
                    print("The true positives: " + str(detection_metrics["true_positives"]), file=f)
                    print("The false negatives: " + str(detection_metrics["false_negatives"]), file=f)
                    print("The false positives: " + str(detection_metrics["false_positives"]), file=f)
            #Since not all clients are used for fedemnist, we write all the selected clients for the round to the output file
            elif(selectedDataset == "fedemnist"):
                with open(filename, "a") as f:
                    print("Using minmax", file=f)
                    print("All selected clients: {}".format(all_clients), file=f)
                    print("The predicted malicious clients: {}".format(predicted_malicious), file=f)
                    print("The true positives: " + str(detection_metrics["true_positives"]), file=f)
                    print("The false negatives: " + str(detection_metrics["false_negatives"]), file=f)
                    print("The false positives: " + str(detection_metrics["false_positives"]), file=f)
                    #print(predicted_malicious, file=f)

            new_results = []
            for proxy, client in results:
                if(client.metrics["clientID"] not in predicted_malicious):
                    #results.remove((proxy, client))
                    new_results.append((proxy, client))  
                    #print("Keeping client {}".format(str(client.metrics["clientID"])))
                    if(selectedDataset == "cifar10"):
                        with open(filename, "a") as f:
                            pass
                            #print("Keeping client {}".format(str(client.metrics["clientID"])), file=f)
                    else:
                        with open(filename, "a") as f:
                            pass
                            #print("Keeping client {}".format(str(client.metrics["clientID"])), file=f)
            
            newClientIDs = []
            for proxy, client in new_results:
                newClientIDs.append(client.metrics["clientID"])
                #print(client.metrics["clientID"])
            print("Clients in the new results: {}".format(str(sorted(newClientIDs))))
            results = new_results
        
        #V2 uses lof and kmeans. Both methods are passed all the clients each round.  Lof uses minmax and kmeans uses tsne features
        if(ourDetectV2):
            df = pd.DataFrame(update_dict)
            #print(df)
            K = len(df.columns)
            detection_slice = df.tail(10).reset_index(drop=True)
            #for column in detection_slice.columns:
                #print(column)
            #    detection_slice.rename({column: "Client_" + str(column)}, axis=1, inplace=True)
            X1, clients, malicious = our_detection_v2.extract_features_tsne(detection_slice, selectedDataset)
            X2, clients, malicious = our_detection_v2.extract_features_minmax(detection_slice, selectedDataset)
            predicted1 = our_detection_v2.kmeans_clustering(X1, clients)
            print ('kmeans prediciton:', predicted1)
            predicted2 = our_detection_v2.local_outlier_factor(X2, clients, 0.1)
            predicted = np.unique(np.concatenate((predicted1, predicted2), axis=0))
            print ('lof prediction:', predicted2)
            print("Final Predication: ")
            print(predicted)
            #print(type(predicted))

            false_positives = []
            true_positives = []
            false_negatives = []
            for value in predicted:
                if(value < 338):
                    true_positives.append(value)
                else:
                    false_positives.append(value)
            for value in malicious:
                if(value not in predicted):
                    false_negatives.append(value)
            predicted_list = []
            for value in predicted:
                predicted_list.append(value)
            client_list = []
            for value in clients:
                client_list.append(value)
            # final results are written to output file
            with open(filename, "a") as f:
                print("All selected clients: {}".format(sorted(client_list)), file=f)
                print("The predicted malicious clients: {}".format(sorted(predicted_list)), file=f)
                print("The true positives: {}".format(sorted(true_positives)), file=f)
                print("The false negatives: {}".format(sorted(false_negatives)), file=f)
                print("The false positives: {}".format(sorted(false_positives)), file=f)
                #our_detection_v2.evaluate(clients, malicious, predicted, f, server_round)

            new_results = []
            for proxy, client in results:
                if(client.metrics["clientID"] not in predicted):
                    #print("Client {} is not marked as malicious".format(client.metrics["clientID"]))
                    new_results.append((proxy, client))

            newClientIDs = []
            for proxy, client in new_results:
                newClientIDs.append(client.metrics["clientID"])
            results = new_results

        #V3 uses more of a pipline structure. All clients are sent to lof, the predicted malicious clients are removed, then the remaining clients are sent to kmeans
        #lof still uses minmax features and kmeans still uses tsne features
        if(V3):
            df = pd.DataFrame(update_dict)
            K = len(df.columns)
            detection_slice = df.tail(10).reset_index(drop=True)
            #used for graphing clusters
            #saved_csv = detection_slice.to_csv('10_rounds_tests/testsForPaperGraphs/fmnist/fmnist_Test2_Round{}_client_models.csv'.format(str(server_round)), index=False)
            #for column in detection_slice.columns:
                #print(column)
            #    detection_slice.rename({column: "Client_" + str(column)}, axis=1, inplace=True)

            #used to run lof only
            if(lofOnly):
                X1, clients1, malicious = our_detection_v3.extract_features_minmax(detection_slice, selectedDataset)
                lof_predicted_benign, lof_predicted_malicious = our_detection_v3.local_outlier_factor(X1, clients1, 0.1)
                print ('lof prediction benign:', sorted(lof_predicted_benign))

                clients = clients1
                predicted_malicious = lof_predicted_malicious

            #used to run lof and kmeans
            else:
                X1, clients1, malicious = our_detection_v3.extract_features_minmax(detection_slice, selectedDataset)
                lof_predicted_benign, lof_predicted_malicious = our_detection_v3.local_outlier_factor(X1, clients1, 0.1)
                print ('lof prediction benign:', sorted(lof_predicted_benign))
                
                filtered_dataset = detection_slice.filter(items=list(map(int, lof_predicted_benign)))
                X2, clients2, malicious = our_detection_v3.extract_features_tsne(filtered_dataset, selectedDataset)
                kmeans_predicted_malicious = our_detection_v3.kmeans_clustering(X2, clients2)
                print ('kmeans malicious prediction:', sorted(kmeans_predicted_malicious))
            
                clients = np.unique(np.concatenate((clients1, clients2), axis=0))
                predicted_malicious = np.unique(np.concatenate((lof_predicted_malicious, kmeans_predicted_malicious), axis=0))

            false_positives = []
            true_positives = []
            false_negatives = []
            for value in predicted_malicious:
                if(value < 4):
                    true_positives.append(value)
                else:
                    false_positives.append(value)
            for value in malicious:
                if(value not in predicted_malicious):
                    false_negatives.append(value)
            predicted_list = []
            for value in predicted_malicious:
                predicted_list.append(value)
            client_list = []
            for value in clients:
                client_list.append(value)
            # final results are written to output file
            with open(filename, "a") as f:
                print("All selected clients: {}".format(sorted(client_list)), file=f)
                print("The predicted malicious clients: {}".format(sorted(predicted_list)), file=f)
                print("The true positives: {}".format(sorted(true_positives)), file=f)
                print("The false negatives: {}".format(sorted(false_negatives)), file=f)
                print("The false positives: {}".format(sorted(false_positives)), file=f)
                #our_detection_v2.evaluate(clients, malicious, predicted, f, server_round)

            new_results = []
            for proxy, client in results:
                if(client.metrics["clientID"] not in predicted_malicious):
                    #print("Client {} is not marked as malicious".format(client.metrics["clientID"]))
                    new_results.append((proxy, client))

            newClientIDs = []
            for proxy, client in new_results:
                newClientIDs.append(client.metrics["clientID"])
            results = new_results

        #uses perfect knowledge to remove all malicious clients every round
        if perfect:
            new_results = []
            selectedClients = []
            for proxy, client in results:
                selectedClients.append(client.metrics["clientID"])
                if(client.metrics["clientID"] >= 338):
                    #print("Client {} is not marked as malicious".format(client.metrics["clientID"]))
                    new_results.append((proxy, client))
            
            newClientIDs = []
            for proxy, client in new_results:
                newClientIDs.append(client.metrics["clientID"])
            with open(filename, "a") as f:
                print("All selected clients: {}".format(sorted(selectedClients)), file=f)
                print("The remaining clients: {}".format(sorted(newClientIDs)), file=f)

            results = new_results

        #uses perfect knowledge to remove all malicious clients each round and keep 5 benign clients
        if perfectSmall:
            new_results = []
            selectedClients = []
            for proxy, client in results:
                selectedClients.append(client.metrics["clientID"])
                if(client.metrics["clientID"] >= 338):
                    #print("Client {} is not marked as malicious".format(client.metrics["clientID"]))
                    if(len(new_results) < 5):
                        new_results.append((proxy, client))           

            newClientIDs = []
            for proxy, client in new_results:
                newClientIDs.append(client.metrics["clientID"])
            with open(filename, "a") as f:
                print("All selected clients: {}".format(sorted(selectedClients)), file=f)
                print("The remaining clients: {}".format(sorted(newClientIDs)), file=f)

            results = new_results

        #uses perfect  knowledge to remove all malicious clients every round but injects poisoned clients on specific rounds
        #the rounds to inject poison are hard coded to be rounds 3 and 4
        if perfectPoison:
            benign_counter = 0
            #We want to poison the model on round 3
            if(server_round == 3 or server_round == 4):
                print("Adding POISONED clients")
                new_results = []
                selectedClients = []
                poison_counter = 0
                for proxy, client in results:
                    selectedClients.append(client.metrics["clientID"])
                    #we want to add 5 malicious clients and 20 benign clients 
                    if(poison_counter < 5):
                        if(client.metrics["clientID"] < 338):
                            new_results.append((proxy, client))
                            poison_counter += 1
                    if(benign_counter < 10):
                        if(client.metrics["clientID"] >= 338):
                            new_results.append((proxy, client))
                            benign_counter += 1
            #if it is not round 3, we use perfect detection to keep 25 benign clients per round
            else:
                new_results = []
                selectedClients = []
                for proxy, client in results:
                    selectedClients.append(client.metrics["clientID"])
                    if(benign_counter < 15):
                        if(client.metrics["clientID"] >= 338):
                            new_results.append((proxy, client))
                            benign_counter += 1

            newClientIDs = []
            for proxy, client in new_results:
                newClientIDs.append(client.metrics["clientID"])
            with open(filename, "a") as f:
                print("All selected clients: {}".format(sorted(selectedClients)), file=f)
                print("The remaining clients: {}".format(sorted(newClientIDs)), file=f)

            results = new_results
        
        #This method uses V3 detection then uses RLR on what is left
        if hybrid:
            df = pd.DataFrame(update_dict)
            #print(df)
            K = len(df.columns)
            detection_slice = df.tail(10).reset_index(drop=True)
            #for column in detection_slice.columns:
                #print(column)
            #    detection_slice.rename({column: "Client_" + str(column)}, axis=1, inplace=True)
            X1, clients1, malicious = our_detection_v3.extract_features_minmax(detection_slice, selectedDataset)
            lof_predicted_benign, lof_predicted_malicious = our_detection_v3.local_outlier_factor(X1, clients1, 0.1)
            print ('lof prediction benign:', sorted(lof_predicted_benign))
            
            filtered_dataset = detection_slice.filter(items=list(map(int, lof_predicted_benign)))
            X2, clients2, malicious = our_detection_v3.extract_features_tsne(filtered_dataset, selectedDataset)
            kmeans_predicted_malicious = our_detection_v3.kmeans_clustering(X2, clients2)
            print ('kmeans malicious prediction:', sorted(kmeans_predicted_malicious))
            
            clients = np.unique(np.concatenate((clients1, clients2), axis=0))

            #print(type(predicted))
            predicted_malicious = np.unique(np.concatenate((lof_predicted_malicious, kmeans_predicted_malicious), axis=0))

            false_positives = []
            true_positives = []
            false_negatives = []
            for value in predicted_malicious:
                if(value < 338):
                    true_positives.append(value)
                else:
                    false_positives.append(value)
            for value in malicious:
                if(value not in predicted_malicious):
                    false_negatives.append(value)
            predicted_list = []
            for value in predicted_malicious:
                predicted_list.append(value)
            client_list = []
            for value in clients:
                client_list.append(value)
            # final results are written to output file
            with open(filename, "a") as f:
                print("All selected clients: {}".format(sorted(client_list)), file=f)
                print("The predicted malicious clients: {}".format(sorted(predicted_list)), file=f)
                print("The true positives: {}".format(sorted(true_positives)), file=f)
                print("The false negatives: {}".format(sorted(false_negatives)), file=f)
                print("The false positives: {}".format(sorted(false_positives)), file=f)
                #our_detection_v2.evaluate(clients, malicious, predicted, f, server_round)

            new_results = []
            for proxy, client in results:
                if(client.metrics["clientID"] not in predicted_malicious):
                    #print("Client {} is not marked as malicious".format(client.metrics["clientID"]))
                    new_results.append((proxy, client))

            #Remove the clients that were filtered out from the update dictionary
            new_update_dict = {}
            for proxy, client in new_results:
                for key, value in update_dict.items():
                    if(client.metrics["clientID"] == key):
                        new_update_dict[key] = value

            newClientIDs = []
            for proxy, client in new_results:
                newClientIDs.append(client.metrics["clientID"])

            lr_vector = compute_robustLR(new_update_dict, len(new_update_dict.keys())*.25)
            #lr_vector = compute_robustLR(new_update_dict, 3)
            print("LR vector based on hybrid method:")
            print(lr_vector)

            results = new_results

        #This method uses lof detection and then uses RLR on what is left
        if lofHybrid:
            df = pd.DataFrame(update_dict)
            K = len(df.columns)
            #full_model = df.to_csv('Round1_fmnist_full_client_models.csv', index=False)
            detection_slice = df.tail(10).reset_index(drop=True)

            if(justUTD):
                predicted_malicious = []
            else:

                #used to run only lof
                X1, clients1, malicious = our_detection_v3.extract_features_minmax(detection_slice, selectedDataset)
                lof_predicted_benign, lof_predicted_malicious = our_detection_v3.local_outlier_factor(X1, clients1, 0.1)
                #lof_predicted_benign, lof_predicted_malicious = our_detection_v3.kmeans_clustering(X1, clients1)
                print ('lof prediction malicious:', sorted(lof_predicted_malicious))

                clients = clients1
                predicted_malicious = lof_predicted_malicious

                false_positives = []
                true_positives = []
                false_negatives = []
                for value in predicted_malicious:
                    if(value < 1):
                        true_positives.append(value)
                    else:
                        false_positives.append(value)
                for value in malicious:
                    if(value not in predicted_malicious):
                        false_negatives.append(value)
                predicted_list = []
                for value in predicted_malicious:
                    predicted_list.append(value)
                client_list = []
                for value in clients:
                    client_list.append(value)
                # final results are written to output file
                with open(filename, "a") as f:
                    print("All selected clients: {}".format(sorted(client_list)), file=f)
                    print("The predicted malicious clients: {}".format(sorted(predicted_list)), file=f)
                    print("The true positives: {}".format(sorted(true_positives)), file=f)
                    print("The false negatives: {}".format(sorted(false_negatives)), file=f)
                    print("The false positives: {}".format(sorted(false_positives)), file=f)

            new_results = []
            for proxy, client in results:
                if(client.metrics["clientID"] not in predicted_malicious):
                    #print("Client {} is not marked as malicious".format(client.metrics["clientID"]))
                    new_results.append((proxy, client))

            #Remove the clients that were filtered out from the update dictionary
            new_update_dict = {}
            for proxy, client in new_results:
                for key, value in update_dict.items():
                    if(client.metrics["clientID"] == key):
                        new_update_dict[key] = value

            newClientIDs = []
            for proxy, client in new_results:
                newClientIDs.append(client.metrics["clientID"])

            #compute RLR based on the updated update dictionary
            lr_vector = compute_robustLR(new_update_dict, len(new_update_dict.keys())*.25)
            print("LR vector based on lof hybrid method:")
            print(lr_vector)

            results = new_results

        if kmeansHybrid:
            df = pd.DataFrame(update_dict)
            K = len(df.columns)
            #full_model = df.to_csv('Round1_fmnist_full_client_models.csv', index=False)
            detection_slice = df.tail(10).reset_index(drop=True)
            #round_values = detection_slice.to_csv('testsForPaperGraphs/fmnist_testing/cluster_visualization/Round{}_kmeans_detection_slice.csv'.format(server_round), index=False)

            #used to run only lof
            X1, clients1, malicious = our_detection_v3.extract_features_minmax(detection_slice, selectedDataset)
            kmeans_predicted_malicious = our_detection_v3.kmeans_clustering(X1, clients1)
            print ('kmeans prediction malicious:', sorted(kmeans_predicted_malicious))

            clients = clients1
            predicted_malicious = kmeans_predicted_malicious

            false_positives = []
            true_positives = []
            false_negatives = []
            for value in predicted_malicious:
                if(value < 1):
                    true_positives.append(value)
                else:
                    false_positives.append(value)
            for value in malicious:
                if(value not in predicted_malicious):
                    false_negatives.append(value)
            predicted_list = []
            for value in predicted_malicious:
                predicted_list.append(value)
            client_list = []
            for value in clients:
                client_list.append(value)
            # final results are written to output file
            with open(filename, "a") as f:
                print("All selected clients: {}".format(sorted(client_list)), file=f)
                print("The predicted malicious clients: {}".format(sorted(predicted_list)), file=f)
                print("The true positives: {}".format(sorted(true_positives)), file=f)
                print("The false negatives: {}".format(sorted(false_negatives)), file=f)
                print("The false positives: {}".format(sorted(false_positives)), file=f)

            new_results = []
            for proxy, client in results:
                if(client.metrics["clientID"] not in predicted_malicious):
                    #print("Client {} is not marked as malicious".format(client.metrics["clientID"]))
                    new_results.append((proxy, client))

            #Remove the clients that were filtered out from the update dictionary
            new_update_dict = {}
            for proxy, client in new_results:
                for key, value in update_dict.items():
                    if(client.metrics["clientID"] == key):
                        new_update_dict[key] = value

            newClientIDs = []
            for proxy, client in new_results:
                newClientIDs.append(client.metrics["clientID"])

            #compute RLR based on the updated update dictionary
            lr_vector = compute_robustLR(new_update_dict, len(new_update_dict.keys())*.25)
            print("LR vector based on kmeans hybrid method:")
            print(lr_vector)

            results = new_results

        if(percentile):
            df = pd.DataFrame(update_dict)
            K = len(df.columns)
            #full_model = df.to_csv('Round1_fmnist_full_client_models.csv', index=False)
            detection_slice = df.tail(10).reset_index(drop=True)
            predicted_benign, predicted_malicious, clients, malicious = percentileDetection.percentileDetection(detection_slice, selectedDataset)
            
            false_positives = []
            true_positives = []
            false_negatives = []
            for value in predicted_malicious:
                if(value < 1):
                    true_positives.append(value)
                else:
                    false_positives.append(value)
            for value in malicious:
                if(value not in predicted_malicious):
                    false_negatives.append(value)
            predicted_list = []
            for value in predicted_malicious:
                predicted_list.append(value)
            client_list = []
            for value in clients:
                client_list.append(value)
            # final results are written to output file
            with open(filename, "a") as f:
                print("All selected clients: {}".format(sorted(client_list)), file=f)
                print("The predicted malicious clients: {}".format(sorted(predicted_list)), file=f)
                print("The true positives: {}".format(sorted(true_positives)), file=f)
                print("The false negatives: {}".format(sorted(false_negatives)), file=f)
                print("The false positives: {}".format(sorted(false_positives)), file=f)

            new_results = []
            for proxy, client in results:
                if(client.metrics["clientID"] not in predicted_malicious):
                    #print("Client {} is not marked as malicious".format(client.metrics["clientID"]))
                    new_results.append((proxy, client))

            #Remove the clients that were filtered out from the update dictionary
            new_update_dict = {}
            for proxy, client in new_results:
                for key, value in update_dict.items():
                    if(client.metrics["clientID"] == key):
                        new_update_dict[key] = value

            newClientIDs = []
            for proxy, client in new_results:
                newClientIDs.append(client.metrics["clientID"])

            results = new_results

        if(percentileHybrid):
            df = pd.DataFrame(update_dict)
            K = len(df.columns)
            #full_model = df.to_csv('Round1_fmnist_full_client_models.csv', index=False)
            detection_slice = df.tail(10).reset_index(drop=True)
            #round_values = detection_slice.to_csv('testsForPaperGraphs/fmnist_testing/percentile/cluster_visualization/test1/Round{}_kmeans_detection_slice.csv'.format(server_round), index=False)
            predicted_benign, predicted_malicious, clients, malicious = percentileDetection.percentileDetection(detection_slice, selectedDataset)
            
            false_positives = []
            true_positives = []
            false_negatives = []
            for value in predicted_malicious:
                if(value < 1):
                    true_positives.append(value)
                else:
                    false_positives.append(value)
            for value in malicious:
                if(value not in predicted_malicious):
                    false_negatives.append(value)
            predicted_list = []
            for value in predicted_malicious:
                predicted_list.append(value)
            client_list = []
            for value in clients:
                client_list.append(value)
            # final results are written to output file
            with open(filename, "a") as f:
                print("All selected clients: {}".format(sorted(client_list)), file=f)
                print("The predicted malicious clients: {}".format(sorted(predicted_list)), file=f)
                print("The true positives: {}".format(sorted(true_positives)), file=f)
                print("The false negatives: {}".format(sorted(false_negatives)), file=f)
                print("The false positives: {}".format(sorted(false_positives)), file=f)

            new_results = []
            for proxy, client in results:
                if(client.metrics["clientID"] not in predicted_malicious):
                    #print("Client {} is not marked as malicious".format(client.metrics["clientID"]))
                    new_results.append((proxy, client))

            #Remove the clients that were filtered out from the update dictionary
            new_update_dict = {}
            for proxy, client in new_results:
                for key, value in update_dict.items():
                    if(client.metrics["clientID"] == key):
                        new_update_dict[key] = value

            newClientIDs = []
            for proxy, client in new_results:
                newClientIDs.append(client.metrics["clientID"])

            #compute RLR based on the updated update dictionary
            lr_vector = compute_robustLR(new_update_dict, len(new_update_dict.keys())*.25)
            print("LR vector based on kmeans hybrid method:")
            print(lr_vector)

            results = new_results

        #UTD RLR method
        if UTDDetect:
            print("RUNNING UTD DETECTION")
            if(selectedDataset == "cifar10"):
                with open(filename, "a") as f:
                    print("RUNNING UTD DETECTION", file=f)
            else:
                with open(filename, "a") as f:
                    print("RUNNING UTD DETECTION", file=f)
            lr_vector = compute_robustLR(update_dict, 8)
            print(lr_vector)
        
        #interpretation of the aggregate.py flower code
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        num_examples_total = sum([num_examples for _, num_examples in weights_results])
        weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in weights_results
        ]
      
        #compute average weights of each layer
        weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
        ]

        #load the correct model
        if selectedDataset == "cifar10":
            model = utils.Net()
        else:
            model = utils.CNN_MNIST()

        params_dict = zip(model.state_dict().keys(), weights_prime)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=False)
        primeParams = parameters_to_vector(model.parameters()).detach()
        
        #apply RLR if it is being used
        if(UTDDetect or hybrid or lofHybrid or kmeansHybrid or percentileHybrid):
            finalParams = primeParams * lr_vector
        else:
            finalParams = primeParams
        
        #store the final weights back into the model
        vector_to_parameters(finalParams, model.parameters())
        #retrieve the final weights (this converts the params from the UTD format to flower format)
        weights_prime = utils.get_model_params(model)

        #metric stuff
        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["train_accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]
        poisonAccuracies = [r.metrics["poison_accuracy"] * r.num_examples for _, r in results]

        print("Number of clients after removing some: {}".format(str(len(results))))
        if(selectedDataset == "cifar10"):
            with open(filename, "a") as f:
                print("Number of clients after removing some: {}".format(str(len(results))), file=f)
        else:
            with open(filename, "a") as f:
                print("Number of clients after removing some: {}".format(str(len(results))), file=f)
        
        # Aggregate and print custom metric
        if(sum(examples) > 0):
            aggregated_accuracy = sum(accuracies) / sum(examples)
            print(f"Round {server_round} accuracy aggregated from client fit results: {aggregated_accuracy}")

            aggregated_poison_accuracy = sum(poisonAccuracies) / sum(examples)
            print(f"Round {server_round} poison accuracy aggregated from client fit results: {aggregated_poison_accuracy}")
        else:
            aggregated_accuracy = 0

        #Save the aggregated client training accuracies to the output file
        if(selectedDataset == "cifar10"):
            with open(filename, "a") as f:
                print("Round {} aggregated training accuracy: {}".format(str(server_round), aggregated_accuracy), file=f)
                print("Round {} aggregated poison accuracy: {}".format(str(server_round), aggregated_poison_accuracy), file=f)
        else:
            with open(filename, "a") as f:
                print("Round {} aggregated training accuracy: {}".format(str(server_round), aggregated_accuracy), file=f)
                print("Round {} aggregated poison accuracy: {}".format(str(server_round), aggregated_poison_accuracy), file=f)

        # Return aggregated model paramters and other metrics (i.e., aggregated accuracy)
        return ndarrays_to_parameters(weights_prime), {"accuracy": aggregated_accuracy}
        #return model.parameters(), {"accuracy": aggregated_accuracy}
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation accuracy using weighted average."""

        if not results:
            return None, {}

        # Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]
        #poisonAccuracies = [r.metrics["poison_accuracy"] * r.num_examples for _, r in results]

        # Aggregate and print custom metric
        if(sum(examples) > 0):
            aggregated_accuracy = sum(accuracies) / sum(examples)
            print(f"Round {server_round} accuracy aggregated from client eval results: {aggregated_accuracy}")
        else:
            aggregated_accuracy = 0

        #aggregated_poison_accuracy = sum(poisonAccuracies) / sum(examples)
        #print(f"Round {server_round} poison accuracy aggregated from client fit results: {aggregated_poison_accuracy}")

        # Return aggregated loss and metrics (i.e., aggregated accuracy)
        return aggregated_loss, {"accuracy": aggregated_accuracy}

def compute_robustLR(agent_updates_dict, threshold):
        print("Threshold: " + str(threshold))
        agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]  
        sm_of_signs = torch.abs(sum(agent_updates_sign))
        
        sm_of_signs[sm_of_signs < threshold] = -1
        sm_of_signs[sm_of_signs >= threshold] = 1                                           
        return sm_of_signs

def main():
    """Load model for
    1. server-side parameter initialization
    2. server-side parameter evaluation
    """

    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--toy",
        type=bool,
        default=False,
        required=False,
        help="Set to true to use only 10 datasamples for validation. \
            Useful for testing purposes. Default: False",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="cifar10",
        required=False,
        help="Used to select the dataset to train on"
    )
    parser.add_argument(
        "--UTDDetect",
        type=bool,
        default=False,
        required=False,
        help="Toggle to enable or disable UTD's poisoning detection/mitigation"
    )
    parser.add_argument(
        "--ourDetect",
        type=bool,
        default=False,
        required=False,
        help="Toggle to enable or disable our poisoning detection/mitigation"
    )
    parser.add_argument(
        "--ourDetectV2",
        type=bool,
        default=False,
        required=False,
        help="Toggle to enable or disable our poisoning detection/mitigation V2"
    )
    parser.add_argument(
        "--V3",
        type=bool,
        default=False,
        required=False,
        help="Toggle to enable or disable our poisoning detection/mitigation V3"
    )
    parser.add_argument(
        "--lofOnly",
        type=bool,
        default=False,
        required=False,
        help="Toggle to only run lof when using V3 detection"
    )
    parser.add_argument(
        "--perfect",
        type=bool,
        default=False,
        required=False,
        help="Toggle to enable perfect detection"
    )
    parser.add_argument(
        "--perfectSmall",
        type=bool,
        default=False,
        required=False,
        help="Toggle to enable perfect detection, but only include 5 benign clients per round"
    )
    parser.add_argument(
        "--perfectPoison",
        type=bool,
        default=False,
        required=False,
        help="Toggle to enable perfect detection, but allow poisoning to happen for 1 round"
    )
    parser.add_argument(
        "--hybrid",
        type=bool,
        default=False,
        required=False,
        help="Toggle to enable V3 detection followed by UTD RLR"
    )
    parser.add_argument(
        "--lofHybrid",
        type=bool,
        default=False,
        required=False,
        help="Toggle to enable lof detection followed by UTD RLR"
    )
    parser.add_argument(
        "--kmeansHybrid",
        type=bool,
        default=False,
        required=False,
        help="Toggle to enable kmeans detection followed by UTD RLR"
    )
    parser.add_argument(
        "--kmeansOnly",
        type=bool,
        default=False,
        required=False,
        help="Toggle to enable just kmeans detection"
    )
    parser.add_argument(
        "--justUTD",
        type=bool,
        default=False,
        required=False,
        help="Toggle to remove the lof part of the lofHybrid and just run the UTD method"
    )
    parser.add_argument(
        "--cluster",
        type=str,
        default="kmeans",
        required=False,
        help="The clustering algorithm to use for our detection method"
    )
    parser.add_argument(
        "--percentile",
        type=bool,
        default=False,
        required=False,
        help="Toggle to enable detection by percentile"
    )
    parser.add_argument(
        "--percentileHybrid",
        type=bool,
        default=False,
        required=False,
        help="Toggle to enable detection by percentile with RLR"
    )
    args = parser.parse_args()

    global selectedDataset 
    global UTDDetect
    global ourDetect
    global ourDetectV2
    global V3
    global lofOnly
    global cluster_algorithm
    global filename
    global perfect
    global perfectSmall
    global perfectPoison
    global hybrid
    global lofHybrid
    global justUTD
    global kmeansHybrid
    global kmeansOnly
    global percentile
    global percentileHybrid

    UTDDetect = args.UTDDetect
    ourDetect = args.ourDetect
    ourDetectV2 = args.ourDetectV2
    V3 = args.V3
    lofOnly = args.lofOnly
    perfect = args.perfect
    perfectSmall = args.perfectSmall
    perfectPoison = args.perfectPoison
    hybrid = args.hybrid
    lofHybrid = args.lofHybrid
    justUTD = args.justUTD
    kmeansOnly = args.kmeansOnly
    kmeansHybrid = args.kmeansHybrid
    percentile = args.percentile
    percentileHybrid = args.percentileHybrid

    cluster_algorithm = args.cluster
    selectedDataset = args.data
    
    if(args.data == "cifar10"):
        model = utils.Net()
        #model = utils.CNN_MNIST()
        ct = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        directory = Path("results/cifar10")
        # Create the directory, including parent directories if needed
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Directory '{directory}' created successfully.")
        filename = "cifar10_iid_" + str(ct) + ".txt"
        filepath = directory / filename
        with open(filepath, "w") as f:
            print("Running cifar test", file=f)
    elif(args.data == "fmnist"):
        model = utils.CNN_MNIST()
        ct = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = "fmnist/fmnist_50_percentile_poison_50_test3_10_clients_" + str(ct) + ".txt"
        with open(filename, "w") as f:
            print("Running fmnist test", file=f)
    else:
        model = utils.CNN_MNIST()
        ct = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = "bestMethod/RLR_flower_poison_50_old_threshold_method_500_rounds_test1_fedemnist_33_clients_" + str(ct) + ".txt"
        with open(filename, "w") as f:
            print("Running fedemnist test", file=f)

    print("NUMBER OF PARAMETERS: " + str(len(parameters_to_vector(model.parameters()))))

    model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]

    # Create strategy
    num_agents = 33 if selectedDataset == "fedemnist" else 10
    '''
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=num_agents,
        min_evaluate_clients=num_agents,
        min_available_clients=num_agents,
        evaluate_fn=get_evaluate_fn(model, args.toy, args.data),
        on_fit_config_fn=get_on_fit_config_fn(), #fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
    )
    '''
    strategy = AggregateCustomMetricStrategy(
        min_fit_clients=num_agents,
        min_evaluate_clients=num_agents,
        min_available_clients=num_agents,
        evaluate_fn=get_evaluate_fn(model, args.toy, args.data),
        on_fit_config_fn=get_on_fit_config_fn(), #fit_config,
        on_evaluate_config_fn=evaluate_config,
        server_learning_rate = 1.0,
        server_momentum = 0,
        initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
        
    )
    # Start Flower server for four rounds of federated learning
    fl.server.start_server(
        server_address="10.100.116.10:8080",
        config=fl.server.ServerConfig(num_rounds=100 if selectedDataset == "fedemnist" else 10),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
