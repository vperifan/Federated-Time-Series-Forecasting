from typing import List

import torch


class LSTM(torch.nn.Module):
    def __init__(self, input_dim: int, lstm_hidden_size: int = 128, num_lstm_layers: int = 2, lstm_dropout: float = 0.2,
                 layer_units: List[int] = [128, 64], num_outputs: int = 2, init_weights: bool = True,
                 matrix_rep: bool = False, exogenous_dim: int = 0):
        super(LSTM, self).__init__()

        self.is_matrix = matrix_rep

        activation = torch.nn.ReLU()

        self.hidden_dim = lstm_hidden_size
        self.layer_dim = num_lstm_layers

        self.lstm = torch.nn.LSTM(input_size=input_dim, hidden_size=lstm_hidden_size,
                                  num_layers=num_lstm_layers, batch_first=True, dropout=lstm_dropout)

        if len(layer_units) == 1:
            layers = [torch.nn.Linear(lstm_hidden_size + exogenous_dim, num_outputs)]
        else:
            layers = [torch.nn.Linear(lstm_hidden_size + exogenous_dim, layer_units[0]), activation]  # Input layer
            for i in range(len(layer_units) - 1):  # hidden layers
                layers.append(torch.nn.Linear(layer_units[i], layer_units[i + 1]))
                layers.append(activation)
            layers.append(torch.nn.Linear(layer_units[len(layer_units) - 1], num_outputs))  # Output layer

        self.MLP_layers = torch.nn.Sequential(*layers)
        if init_weights:
            self.MLP_layers.apply(self._init_weights)

    def forward(self, x, exogenous_data=None, device="cpu", y_hist=None):
        if not self.is_matrix:
            x = x.view([x.size(0), -1, x.size(1)])
        else:
            x = x.reshape(x.size(0), x.size(1), x.size(2))

        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)

        out, (_, _) = self.lstm(x, (h0.detach(), c0.detach()))
        out = out[:, -1, :]

        if exogenous_data is not None and self.is_matrix:
            out = torch.cat((out, exogenous_data), dim=1)

        out = self.MLP_layers(out)
        return out

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
