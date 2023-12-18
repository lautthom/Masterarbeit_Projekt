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
    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device, dtype=torch.float32)

            outputs = model(data, device)

            outputs_array = outputs.cpu().numpy()

            predictions = np.where(outputs_array < 0.5, 1, 4)
            labels = labels
    model.train()

    if show_confusion_matrix:
        ConfusionMatrixDisplay.from_predictions(labels, predictions)
        plt.show()

    return accuracy_score(labels, predictions), outputs_array


def run_model(model_type, data_train, labels_train, data_test, labels_test, classes, learning_rate, epochs, batch_size, hidden_state_size=1, num_recurrent_layer=1, kernel_size=11, use_grus=False, show_training=False, show_confusion_matrix=False, show_training_plot=False, finetuning_data=None, finetuning_labels=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    labels_train_copy = labels_train.copy()
    
    labels_train_relabeled = deep_learning_utils.relabel(labels_train_copy, classes)
    
    labels_train_relabeled = np.expand_dims(labels_train_relabeled, axis=1)

    if finetuning_labels is not None:
        finetuning_labels_copy = finetuning_labels.copy()
        finetuning_labels_relabeled = deep_learning_utils.relabel(finetuning_labels_copy, classes)
        finetuning_labels_relabeled = np.expand_dims(finetuning_labels_relabeled, axis=1)

    if model_type == 'rnn':
        model = deep_learning_architectures.RNN(hidden_dim=hidden_state_size, num_layers=num_recurrent_layer, use_grus=use_grus).to(device)
    elif model_type == 'crnn':
        model = deep_learning_architectures.CRNN(hidden_dim=hidden_state_size, num_layers=num_recurrent_layer, use_grus=use_grus).to(device)
        data_train = np.transpose(data_train, (0, 2, 1))
        data_test = np.transpose(data_test, (0, 2, 1))
        if finetuning_data is not None:
            finetuning_data = np.transpose(finetuning_data, (0, 2, 1))
    elif model_type == 'feature_rnn':
        model = deep_learning_architectures.FeatureRNN(hidden_dim=hidden_state_size, num_layers=num_recurrent_layer, use_grus=use_grus).to(device)
    elif model_type == 'cnn':
        model = deep_learning_architectures.CNN(data_train.shape[1], kernel_size=kernel_size).to(device)
        data_train = np.transpose(data_train, (0, 2, 1))
        data_test = np.transpose(data_test, (0, 2, 1))
        if finetuning_data is not None:
            finetuning_data = np.transpose(finetuning_data, (0, 2, 1))
      
    train_dataloader = deep_learning_utils.make_dataloader(data_train, labels_train_relabeled, batch_size)
    test_dataloader = deep_learning_utils.make_dataloader(data_test, labels_test, 40, shuffle=False)

    if finetuning_data is not None and finetuning_labels is not None:
        finetuning_dataloader = deep_learning_utils.make_dataloader(finetuning_data, finetuning_labels_relabeled, batch_size)
    else:
        finetuning_dataloader = train_dataloader

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)  

    train_accuracies = []

    for i in range(epochs):
        loss_epoch = 0

        #delete unnecessary lists        
        predictions_epoch = []
        labels_epoch = []

        for data, labels in train_dataloader:
            data = data.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)

            optimizer.zero_grad()

            outputs = model(data, device)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_epoch += loss.item()

            # shorten with np.where()
            for output, label in zip(outputs, labels):
                prediction = 1 if output >= 0.5 else 0
                predictions_epoch.append(prediction)
                labels_epoch.append(label.item())
 
        train_accuracy = accuracy_score(labels_epoch, predictions_epoch)

        train_accuracies.append(train_accuracy)

        if show_training:
            print(f'Epoch: {i}, Loss: {loss_epoch:.2f}, Train accuracy: {train_accuracy:.3f}')

    optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.001)

    for i in range(epochs):
        loss_epoch = 0

        #delete unnecessary lists        
        predictions_epoch = []
        labels_epoch = []

        for data, labels in finetuning_dataloader:
            data = data.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)

            optimizer.zero_grad()

            outputs = model(data, device)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_epoch += loss.item() 

            #shorten with np.array()
            for output, label in zip(outputs, labels):
                prediction = 1 if output >= 0.5 else 0
                predictions_epoch.append(prediction)
                labels_epoch.append(label.item())
 
        train_accuracy = accuracy_score(labels_epoch, predictions_epoch)

        train_accuracies.append(train_accuracy)

        if show_training:
            print(f'Epoch: {i}, Loss: {loss_epoch:.2f}, Train accuracy: {train_accuracy:.3f}')
    
    # extend training plot to include loss or deprecate
    if show_training_plot:
        deep_learning_utils.make_training_plot(train_accuracies)
    
    accuracy, outputs_evaluation = run_evaluation(model, test_dataloader, device, show_confusion_matrix=show_confusion_matrix)

    return accuracy, outputs_evaluation, model

def run_loaded_model(model, model_type, data_test, labels_test, classes, show_confusion_matrix=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 

    if model_type == 'cnn' or model_type == 'crnn':
        data_test = np.transpose(data_test, (0, 2, 1))
      
    test_dataloader = deep_learning_utils.make_dataloader(data_test, labels_test, 40, shuffle=False)

    accuracy, outputs_evaluation = run_evaluation(model, test_dataloader, device, show_confusion_matrix=show_confusion_matrix)

    return accuracy, outputs_evaluation



