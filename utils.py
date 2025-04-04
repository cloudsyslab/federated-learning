import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.cuda.amp import autocast
import torch.nn as nn
import torch.nn.functional as F
import cv2
import random
from collections import defaultdict
import numpy as np
from math import floor
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CNN_MNIST(nn.Module):
    def __init__(self):
        super(CNN_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3))
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.drop1 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.max_pool(x)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = self.drop1(x)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = self.fc2(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,   64,  3)
        self.conv2 = nn.Conv2d(64,  128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc2 = nn.Linear(128, 256)
        self.drop3 = nn.Dropout2d(p=0.5)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = self.drop1(x)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.drop3(x)
        x = self.fc3(x)
        return x

class H5Dataset(Dataset):
    def __init__(self, dataset, client_id):
        self.targets = torch.LongTensor(dataset[client_id]['label'])
        self.inputs = torch.Tensor(dataset[client_id]['pixels'])
        shape = self.inputs.shape
        self.inputs = self.inputs.view(shape[0], 1, shape[1], shape[2])
        
    def classes(self):
        return torch.unique(self.targets)
    
    def __add__(self, other): 
        self.targets = torch.cat( (self.targets, other.targets), 0)
        self.inputs = torch.cat( (self.inputs, other.inputs), 0)
        return self
    
    def to(self, device):
        self.targets = self.targets.to(device)
        self.inputs = self.inputs.to(device)
        
        
    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, item):
        inp, target = self.inputs[item], self.targets[item]
        return inp, target

def load_data(data):
    if(data == "cifar10"):
        """Load CIFAR-10 (training and test set)."""
        transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
            ])
        transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
            ])
        
        #transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.2860], std=[0.3530])])
        #transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.2860], std=[0.3530])])

        trainset = CIFAR10("./dataset", train=True, download=True, transform=transform_train)
        testset = CIFAR10("./dataset", train=False, download=True, transform=transform_test)
        trainset.targets, testset.targets = torch.LongTensor(trainset.targets), torch.LongTensor(testset.targets)

    elif(data == "fmnist"):
        transform =  transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.2860], std=[0.3530])])
        trainset = datasets.FashionMNIST("./dataset", train=True, download=True, transform=transform)
        testset = datasets.FashionMNIST("./dataset", train=False, download=True, transform=transform)

    elif(data == "fedemnist"):
        train_dir = './dataset/Fed_EMNIST/fed_emnist_all_trainset.pt'
        test_dir = './dataset/Fed_EMNIST/fed_emnist_all_valset.pt'
        trainset = torch.load(train_dir)
        testset = torch.load(test_dir)

    num_examples = {"trainset": len(trainset), "testset": len(testset)}
    return trainset, testset, num_examples


def load_partition(idx: int):
    """Load 1/10th of the training and test data to simulate a partition."""
    assert idx in range(10)
    trainset, testset, num_examples = load_data("cifar10")
    #return (trainset, testset)
    n_train = int(num_examples["trainset"] / 40)
    n_test = int(num_examples["testset"] / 40)

    train_parition = torch.utils.data.Subset(
        trainset, range(idx * n_train, (idx + 1) * n_train)
    )
    test_parition = torch.utils.data.Subset(
        testset, range(idx * n_test, (idx + 1) * n_test)
    )
    return (train_parition, test_parition)

class DatasetSplit(Dataset):
    """ An abstract Dataset class wrapped around Pytorch Dataset class """
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = idxs
        self.targets = torch.Tensor([self.dataset.targets[idx] for idx in idxs])

    def classes(self):
        return torch.unique(self.targets)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        inp, target = self.dataset[self.idxs[item]]
        return inp, target

def distribute_data(dataset, num_agents=10, n_classes=10, class_per_agent=10):
    #if args.num_agents == 1:
    #    return {0:range(len(dataset))}

    def chunker_list(seq, size):
        return [seq[i::size] for i in range(size)]
    
    # sort labels
    labels_sorted = dataset.targets.sort()
    # create a list of pairs (index, label), i.e., at index we have an instance of  label
    class_by_labels = list(zip(labels_sorted.values.tolist(), labels_sorted.indices.tolist()))
    # convert list to a dictionary, e.g., at labels_dict[0], we have indexes for class 0
    labels_dict = defaultdict(list)
    for k, v in class_by_labels:
        labels_dict[k].append(v)
        
    # split indexes to shards
    shard_size = len(dataset) // (num_agents * class_per_agent)
    slice_size = (len(dataset) // n_classes) // shard_size    
    for k, v in labels_dict.items():
        labels_dict[k] = chunker_list(v, slice_size)
           
    # distribute shards to users
    dict_users = defaultdict(list)
    for user_idx in range(num_agents):
        class_ctr = 0
        for j in range(0, n_classes):
            if class_ctr == class_per_agent:
                    break
            elif len(labels_dict[j]) > 0:
                dict_users[user_idx] += labels_dict[j][0]
                del labels_dict[j%n_classes][0]
                class_ctr+=1

    return dict_users


def poison_dataset(dataset, selectedDataset, data_idxs=None, agent_idx=-1, poison_all=False, pattern='plus'):
    #target of 5 is hard coded for now
    #print("POISONING {}".format(selectedDataset))
    all_idxs = (dataset.targets == 5).nonzero().flatten().tolist()
    if data_idxs != None:
        all_idxs = list(set(all_idxs).intersection(data_idxs))

    poison_frac = 1 if poison_all else 0.5
    #poison_frac = 0.5
    #print("Poinson fraction: " + str(poison_frac))
    poison_idxs = random.sample(all_idxs, floor(poison_frac*len(all_idxs)))
    #print("Poisoning {} images".format(len(poison_idxs)))
    for idx in poison_idxs:
        if selectedDataset == 'fedemnist':
            clean_img = dataset.inputs[idx]
        else:
            clean_img = dataset.data[idx]
        #print("pre: " + str(clean_img.shape))
        #test_image = clean_img.transpose(2,1,0)
        #print("post: " + str(test_image.shape))
        #plt.imshow(clean_img)
        #plt.title("test")
        #print(clean_img)
        #Plus pattern is hard coded for now
        #bd_img = add_pattern_bd(clean_img, selectedDataset, pattern_type='plus', agent_idx=agent_idx)
        #Set agent ID to -1 so we can test non-distributed cifar attack
        bd_img = add_pattern_bd(clean_img, selectedDataset, pattern_type=pattern, agent_idx=-1)
        if selectedDataset == 'fedemnist':
            dataset.inputs[idx] = torch.tensor(bd_img)
        else:
            dataset.data[idx] = torch.tensor(bd_img)
        dataset.targets[idx] = 7
    return


def add_pattern_bd(x, dataset='cifar10', pattern_type='square', agent_idx=-1):
    """
    adds a trojan pattern to the image
    """
    x = np.array(x.squeeze())

    # if cifar is selected, we're doing a distributed backdoor attack (i.e., portions of trojan pattern is split between agents, only works for plus)
    if dataset == 'cifar10':
        if pattern_type == 'square':
            # Create a square in the middle of the CIFAR-10 image (32x32)
            start_idx = 10  # starting index
            size = 12  # size of the square
            for i in range(start_idx, start_idx + size):
                for j in range(start_idx, start_idx + size):
                    x[i, j] = 0  # Set pixel to black (0)

        elif pattern_type == 'copyright':
            # Add copyright pattern to CIFAR-10 image
            trojan = cv2.imread('dataset/watermark.png', cv2.IMREAD_GRAYSCALE)
            trojan = cv2.bitwise_not(trojan)  # Invert the image
            trojan = cv2.resize(trojan, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
            #x = np.minimum(x + trojan, 255)  # Add and ensure values don't exceed 255
            # Convert the grayscale watermark to a 3-channel image
            trojan_color = cv2.cvtColor(trojan, cv2.COLOR_GRAY2BGR)
            x = np.minimum(x + trojan_color, 255)
        elif pattern_type == 'apple':
            # Add apple pattern to CIFAR-10 image
            trojan = cv2.imread('dataset/apple.png', cv2.IMREAD_GRAYSCALE)
            trojan = cv2.bitwise_not(trojan)  # Invert the image
            trojan = cv2.resize(trojan, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
            x = np.minimum(x + trojan, 255)  # Add and ensure values don't exceed 255

        elif pattern_type == 'plus':
            start_idx = 5
            size = 5
            # vertical line
            for i in range(start_idx, start_idx+size):
                x[i, start_idx] = 255

            # horizontal line
            for i in range(start_idx-size//2, start_idx+size//2 + 1):
                x[start_idx+size//2, i] = 255

        elif pattern_type == 'plus_distributed':
            # Add a plus pattern (cross) in the CIFAR-10 image
            start_idx = 5
            size = 6
            if agent_idx == -1:
                # Full cross pattern
                for d in range(0, 3):  # RGB channels
                    for i in range(start_idx, start_idx + size + 1):  # Vertical line
                        x[i, start_idx, d] = 0
                    for i in range(start_idx - size // 2, start_idx + size // 2 + 1):  # Horizontal line
                        x[start_idx + size // 2, i, d] = 0
            else:# DBA attack
                #upper part of vertical
                if agent_idx == 0:
                    for d in range(0, 3):
                        for i in range(start_idx, start_idx+(size//2)+1):
                            x[i, start_idx][d] = 0

                #lower part of vertical
                elif agent_idx == 1:
                    for d in range(0, 3):
                        for i in range(start_idx+(size//2)+1, start_idx+size+1):
                            x[i, start_idx][d] = 0

                #left-part of horizontal
                elif agent_idx == 2:
                    for d in range(0, 3):
                        for i in range(start_idx-size//2, start_idx+size//4 + 1):
                            x[start_idx+size//2, i][d] = 0

                #right-part of horizontal
                elif agent_idx == 3:
                    for d in range(0, 3):
                        for i in range(start_idx-size//4+1, start_idx+size//2 + 1):
                            x[start_idx+size//2, i][d] = 0

    elif dataset == 'fmnist':
        if pattern_type == 'square':
            for i in range(21, 26):
                for j in range(21, 26):
                    x[i, j] = 255

        elif pattern_type == 'copyright':
            trojan = cv2.imread('dataset/watermark.png', cv2.IMREAD_GRAYSCALE)
            trojan = cv2.bitwise_not(trojan)
            trojan = cv2.resize(trojan, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
            x = x + trojan

        elif pattern_type == 'apple':
            trojan = cv2.imread('dataset/apple.png', cv2.IMREAD_GRAYSCALE)
            trojan = cv2.bitwise_not(trojan)
            trojan = cv2.resize(trojan, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
            x = x + trojan

        elif pattern_type == 'plus':
            start_idx = 5
            size = 5
            # vertical line
            for i in range(start_idx, start_idx+size):
                x[i, start_idx] = 255

            # horizontal line
            for i in range(start_idx-size//2, start_idx+size//2 + 1):
                x[start_idx+size//2, i] = 255

        elif pattern_type == 'plus_distributed':
            # Add a plus pattern (cross) in the CIFAR-10 image
            start_idx = 5
            size = 6
            if agent_idx == -1:
                # Full cross pattern
                for d in range(0, 3):  # RGB channels
                    for i in range(start_idx, start_idx + size + 1):  # Vertical line
                        x[i, start_idx, d] = 0
                    for i in range(start_idx - size // 2, start_idx + size // 2 + 1):  # Horizontal line
                        x[start_idx + size // 2, i, d] = 0
            else:# DBA attack
                #upper part of vertical
                if agent_idx == 0:
                    for d in range(0, 3):
                        for i in range(start_idx, start_idx+(size//2)+1):
                            x[i, start_idx][d] = 0

                #lower part of vertical
                elif agent_idx == 1:
                    for d in range(0, 3):
                        for i in range(start_idx+(size//2)+1, start_idx+size+1):
                            x[i, start_idx][d] = 0

                #left-part of horizontal
                elif agent_idx == 2:
                    for d in range(0, 3):
                        for i in range(start_idx-size//2, start_idx+size//4 + 1):
                            x[start_idx+size//2, i][d] = 0

                #right-part of horizontal
                elif agent_idx == 3:
                    for d in range(0, 3):
                        for i in range(start_idx-size//4+1, start_idx+size//2 + 1):
                            x[start_idx+size//2, i][d] = 0

    elif dataset == 'fedemnist':
        if pattern_type == 'square':
            for i in range(21, 26):
                for j in range(21, 26):
                    x[i, j] = 0

        elif pattern_type == 'copyright':
            trojan = cv2.imread('../watermark.png', cv2.IMREAD_GRAYSCALE)

        elif pattern_type == 'apple':
            trojan = cv2.imread('../apple.png', cv2.IMREAD_GRAYSCALE)
            trojan = cv2.bitwise_not(trojan)
            trojan = cv2.resize(trojan, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)/255
            x = x - trojan

        elif pattern_type == 'plus':
            start_idx = 8
            size = 5
            # vertical line
            for i in range(start_idx, start_idx+size):
                x[i, start_idx] = 0

            # horizontal line
            for i in range(start_idx-size//2, start_idx+size//2 + 1):
                x[start_idx+size//2, i] = 0

    return x   



def train(net, trainloader, valloader, poinsonedloader, epochs, device: str = "cpu"):
    """Train the network on the training set."""
    print("Starting training...")
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        net.parameters(), lr=0.1, momentum=0.9 #, weight_decay=1e-4
    )
    #scalar = torch.cuda.amp.GradScaler()
    net.train()
    for _ in range(epochs):
        for _, (images, labels) in enumerate(trainloader):
            optimizer.zero_grad()
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            #print("\nnet(images): " + str(net(images).shape))
            #print("labels: " + str(labels.shape) + "\n")
            #with autocast():
            outputs = net(images)
            loss = criterion(outputs, labels)
            #scalar.scale(loss).backward()
            #scalar.unscale_(optimizer)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 10)
            #scalar.step(optimizer)
            optimizer.step()


            #scalar.update()
            

    net.to("cpu")  # move model back to CPU

    print("train eval")
    train_loss, train_acc, train_per_class = test(net, trainloader, None, device)
    print("val eval")
    val_loss, val_acc, val_per_class = test(net, valloader, None, device)
    print("poison eval")
    poison_loss, poison_acc, poison_per_class = test(net, poinsonedloader, None, device)
    #val_loss, val_acc = test(net, trainloader)

    #print("Length of trainset: " + str(len(trainloader.dataset)))
    #print("Length of validation set: " + str(len(valloader.dataset)))
    #print("Length of poison set: " + str(len(poinsonedloader.dataset)))

    results = {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        #"train_accuracy_per_class": train_per_class,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        #"val_accuracy_per_class": val_per_class,
        "poison_loss": poison_loss,
        "poison_accuracy": poison_acc,
        #"poison_accuracy_per_class": poison_per_class,
    }
    return results


def test(net, testloader, steps: int = None, device: str = "cpu"):
    """Validate the network on the entire test set."""
    print("Starting evaluation...")
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    loss, accuracy, per_class_accuracy = get_loss_and_accuracy(net, criterion, testloader, steps, device)
    net.to("cpu")  # move model back to CPU
    return loss, accuracy, per_class_accuracy

def get_loss_and_accuracy(model, criterion, data_loader, steps: int = None, device: str = "cpu"):
    model.eval()
    #correct, loss = 0, 0.0
    total_loss, correctly_labeled_samples = 0, 0
    confusion_matrix = torch.zeros(10, 10)
    #print("\ttest1")
    with torch.no_grad():
        #print("\ttest2")
        for batch_idx, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
          
            #loss += criterion(outputs, labels).item()
            eps = 1e-6
            avg_minibatch_loss = criterion(outputs, labels)
            if(avg_minibatch_loss.isnan()):
                avg_minibatch_loss = eps
            total_loss += avg_minibatch_loss.item()*outputs.shape[0]

            #_, predicted = torch.max(outputs.data, 1)
            _, pred_labels = torch.max(outputs, 1)
            #predicted = predicted.view(-1)
            pred_labels = pred_labels.view(-1)

            #correct += (predicted == labels).sum().item()
            correctly_labeled_samples += torch.sum(torch.eq(pred_labels, labels)).item()
           
            #if steps is not None and batch_idx == steps:
            #    break
            for t, p in zip(labels.view(-1), pred_labels.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
                
    per_class_accuracy = confusion_matrix.diag() / confusion_matrix.sum(1)
    accuracy = correctly_labeled_samples / len(data_loader.dataset)
    avg_loss = total_loss / len(data_loader.dataset)
    #print("\tAvg Loss: {:.3f}".format(avg_loss))
    #print("\tAccuracy: " + str(accuracy))
    #print("\tPer class accuracy: " + str(per_class_accuracy))
    return avg_loss, accuracy, per_class_accuracy



def get_model_params(model):
    """Returns a model's parameters."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]
