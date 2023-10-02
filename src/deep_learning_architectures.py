import torch
import torch.nn as nn


    

class RNN(nn.Module):
    def __init__(self, hidden_dim, num_layers, use_grus):
        super(RNN, self).__init__()
        if not use_grus:
            self.lstm = nn.LSTM(input_size=3, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        else:
            self.gru = nn.GRU(input_size=3, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * num_layers * 2, 10)
        self.output = nn.Linear(10, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.use_grus = use_grus
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x, device):
        h0 = torch.zeros(self.num_layers * 2, x.shape[0], self.hidden_dim).requires_grad_()   
        h0 = h0.to(device)
        
        if not self.use_grus:
            c0 = torch.zeros(self.num_layers * 2, x.shape[0], self.hidden_dim).requires_grad_()
            c0 = c0.to(device)

            x, (h_n, c_n) = self.lstm(x, (h0.detach(), c0.detach()))
        else:
            x, h_n = self.gru(x, h0.detach())       
        
        last_hidden_states = torch.permute(h_n, (1, 0, 2))
        last_hidden_states_flattened = torch.flatten(last_hidden_states, start_dim=1)
        x = self.relu(self.fc(last_hidden_states_flattened))
        x = self.sigmoid(self.output(x))
        return x
    

class CRNN(nn.Module):
    def __init__(self, hidden_dim, num_layers, use_grus):
        super(CRNN, self).__init__()
        self.conv = nn.Conv1d(in_channels=3, out_channels=8, kernel_size=9, padding='same')
        if not use_grus:
            self.lstm = nn.LSTM(input_size=8, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        else:
            self.gru = nn.GRU(input_size=8, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * num_layers * 2, 10)
        self.output = nn.Linear(10, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.use_grus = use_grus
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x, device):
        h0 = torch.zeros(self.num_layers * 2, x.shape[0], self.hidden_dim).requires_grad_()
        h0 = h0.to(device)
        
        x = self.relu(self.conv(x))
        x = torch.permute(x, (0, 2, 1))
        if not self.use_grus:
            c0 = torch.zeros(self.num_layers * 2, x.shape[0], self.hidden_dim).requires_grad_()
            c0 = c0.to(device)

            x, (h_n, c_n) = self.lstm(x, (h0.detach(), c0.detach()))
        else:
            x, h_n = self.gru(x, h0.detach())
        last_hidden_states = torch.permute(h_n, (1, 0, 2))
        last_hidden_states_flattened = torch.flatten(last_hidden_states, start_dim=1)
        x = self.relu(self.fc(last_hidden_states_flattened))
        x = self.sigmoid(self.output(x))
        return x
    

class FeatureRNN(nn.Module):
    def __init__(self, hidden_dim, num_layers, use_grus):
        super(FeatureRNN, self).__init__()
        if not use_grus:
            self.lstm = nn.LSTM(input_size=36, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        else:
            self.gru = nn.GRU(input_size=36, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * num_layers * 2, 10)
        self.output = nn.Linear(10, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.use_grus = use_grus
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x, device):
        h0 = torch.zeros(self.num_layers * 2, x.shape[0], self.hidden_dim).requires_grad_()
        h0 = h0.to(device)

        if not self.use_grus:
            c0 = torch.zeros(self.num_layers * 2, x.shape[0], self.hidden_dim).requires_grad_()
            c0 = c0.to(device)

            x, (h_n, c_n) = self.lstm(x, (h0.detach(), c0.detach()))
        else:
            x, h_n = self.gru(x, h0.detach())
        last_hidden_states = torch.permute(h_n, (1, 0, 2))
        last_hidden_states_flattened = torch.flatten(last_hidden_states, start_dim=1)
        x = self.relu(self.fc(last_hidden_states_flattened))
        x = self.sigmoid(self.output(x))
        return x
    

class CNN(nn.Module):
    def __init__(self, sample_length):
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(in_channels=3, out_channels=12, kernel_size=9, padding='same')
        self.conv2 = nn.Conv1d(in_channels=12, out_channels=24, kernel_size=9, padding='same')

        self.fc = nn.Linear(sample_length * 24, 512)
        self.fc2 = nn.Linear(512, 100)
        self.output = nn.Linear(100, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, device):
        x = self.relu(self.conv(x))
        x = self.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.output(x))
        return x
    