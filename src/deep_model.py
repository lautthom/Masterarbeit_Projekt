import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
import deep_learning_utils
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import deep_learning_architectures
    

def run_evaluation(model, dataloader, device, show_confusion_matrix=False):
    model.eval()

    predictions = []
    labels_evaluation = []
    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device, dtype=torch.float32)

            outputs = model(data, device)

            for output, label in zip(outputs, labels):
                prediction = 1 if output >= 0.5 else 0
                predictions.append(prediction)
                labels_evaluation.append(label.item())
    model.train()

    if show_confusion_matrix:
        ConfusionMatrixDisplay.from_predictions(labels_evaluation, predictions)
        plt.show()

    return accuracy_score(labels_evaluation, predictions)


def run_model(model, data_train, labels_train, data_test, labels_test, classes, learning_rate, epochs, batch_size, show_confusion_matrix=False, show_training_plot=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    labels_train_copy = labels_train.copy()
    labels_test_copy = labels_test.copy()

    labels_train_relabeled = deep_learning_utils.relabel(labels_train_copy, classes)
    labels_test_relabeled = deep_learning_utils.relabel(labels_test_copy, classes)

    labels_train_relabeled = np.expand_dims(labels_train_relabeled, axis=1)
    labels_test = np.expand_dims(labels_test_relabeled, axis=1)

    if model == 'cnn':
        net = deep_learning_architectures.CNN(data_train.shape[1]).to(device)
        data_train = np.transpose(data_train, (0, 2, 1))
        data_test = np.transpose(data_test, (0, 2, 1))
    elif model == 'rnn':
        net = deep_learning_architectures.RNN(hidden_dim=128, num_layers=2).to(device)
    elif model == 'crnn':
        net = deep_learning_architectures.CRNN(hidden_dim=128, num_layers=2).to(device)
        data_train = np.transpose(data_train, (0, 2, 1))
        data_test = np.transpose(data_test, (0, 2, 1))
    elif model == 'feature_rnn':
        net = deep_learning_architectures.FeatureRNN(hidden_dim=128, num_layers=2).to(device)
        
    train_dataloader = deep_learning_utils.make_dataloader(data_train, labels_train_relabeled, batch_size)
    test_dataloader = deep_learning_utils.make_dataloader(data_test, labels_test, batch_size)

    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)  

    train_accuracies = []

    for i in range(epochs):
        loss_epoch = 0
        
        predictions_epoch = []
        labels_epoch = []

        for data, labels in train_dataloader:
            data = data.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)

            optimizer.zero_grad()

            outputs = net(data, device)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_epoch += loss.item() 

            for output, label in zip(outputs, labels):
                prediction = 1 if output >= 0.5 else 0
                predictions_epoch.append(prediction)
                labels_epoch.append(label.item())
 
        train_accuracy = accuracy_score(labels_epoch, predictions_epoch)

        train_accuracies.append(train_accuracy)

        print(f'Epoch: {i}, Loss: {loss_epoch:.2f}, Train accuracy: {train_accuracy:.3f}')
    
    # extend training plot to include loss or deprecate
    if show_training_plot:
        deep_learning_utils.make_training_plot(train_accuracies)
    
    test_accuracy = run_evaluation(net, test_dataloader, device, show_confusion_matrix=show_confusion_matrix)

    return test_accuracy


