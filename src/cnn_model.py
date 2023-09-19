import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class EDA_Dataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y
    
    def __len__(self):
        return len(self.data)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=9, padding='same')
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=9, padding='same')

        self.fc = nn.Linear(8 * 6144, 512)
        self.fc2 = nn.Linear(512, 100)
        self.output = nn.Linear(100, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, device):
        x = self.relu(self.conv(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.fc(x.view(20, -1)))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.output(x))
        return x
    

def run_evaluation(model, dataloader, device, show_confusion_matrix=False):
    model.eval()

    predictions = []
    true_labels = []
    with torch.no_grad():
        for data, label in dataloader:
            data = data.to(device, dtype=torch.float32)

            prediction = model(data, device)

            for entry, true_label in zip(prediction, label):
                prediction = 1 if entry > 0.5 else 0
                predictions.append(prediction)
                true_labels.append(true_label.item())
    model.train()

    if show_confusion_matrix:
        ConfusionMatrixDisplay.from_predictions(true_labels, predictions)
        plt.show()

    return accuracy_score(true_labels, predictions)


def make_training_plot(train_accuracies, eval_accuracies):
    fig, ax = plt.subplots()
    x = np.arange(0, 25)
    ax.plot(x, train_accuracies, label='train_accuracies')
    ax.plot(x, eval_accuracies, label='eval_accuracies')
    ax.legend() 
    plt.show()


def make_dataloader(data, labels):
    dataset = EDA_Dataset(data, labels)
    return DataLoader(dataset, batch_size=20, shuffle=True)


def relabel(label_array):
    label_array[label_array == 1] = 0
    label_array[label_array == 4] = 1
    return label_array


def run_model(data_train, labels_train, data_test, labels_test, show_confusion_matrix=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #print(f'Using {device} device')

    labels_train_copy = labels_train.copy()
    labels_test_copy = labels_test.copy()

    labels_train_relabeled = relabel(labels_train_copy)
    labels_test_relabeled = relabel(labels_test_copy)

    data_train = np.expand_dims(data_train, axis=1)
    labels_train_relabeled = np.expand_dims(labels_train_relabeled, axis=1)
    
    data_test = np.expand_dims(data_test, axis=1)
    labels_test = np.expand_dims(labels_test_relabeled, axis=1)

    data_train, data_eval = np.split(data_train, [int(((len(data_train) // 20) * 0.8)) * 20])
    labels_train_relabeled, labels_eval = np.split(labels_train_relabeled, [int(((len(labels_train_relabeled) // 20) * 0.8)) * 20])

    train_dataloader = make_dataloader(data_train, labels_train_relabeled)
    eval_dataloader = make_dataloader(data_eval, labels_eval)
    test_dataloader = make_dataloader(data_test, labels_test)

    net = NeuralNetwork().to(device)

    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.005)  # reduce lr during epochs

    train_accuracies = []
    eval_accuracies = []

    for i in range(25):
        loss_epoch = 0

        for index, (data, labels) in enumerate(train_dataloader):
            data = data.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)

            optimizer.zero_grad()

            outputs = net(data, device)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_epoch += loss.item() 
            
        train_accuracy = run_evaluation(net, train_dataloader, device) 
        eval_accuracy = run_evaluation(net, eval_dataloader, device)

        train_accuracies.append(train_accuracy)
        eval_accuracies.append(eval_accuracy)
            
        print(f'Epoch: {i}, Loss: {loss_epoch:.2f}, Train accuracy: {train_accuracy:.3f} Eval accuracy: {eval_accuracy:.3f}')
    
    if show_confusion_matrix:
        make_training_plot(train_accuracies, eval_accuracies)
    
    test_accuracy = run_evaluation(net, test_dataloader, device, show_confusion_matrix=show_confusion_matrix)

    return test_accuracy