from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np


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
    

def make_dataloader(data, labels, batch_size, shuffle=True):
    dataset = EDA_Dataset(data, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def relabel(label_array, classes):
    label_array[label_array == classes[0]] = 0
    label_array[label_array == classes[1]] = 1
    return label_array


def make_training_plot(train_accuracies):
    fig, ax = plt.subplots()
    x = np.arange(0, 25)
    ax.plot(x, train_accuracies, label='train_accuracies')
    ax.legend() 
    plt.show()