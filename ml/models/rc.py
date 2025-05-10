from typing import List
import torch
class ReservoirLayer(torch.nn.Module):
    def __init__(self, input_dim: int, reservoir_size: int, sparsity: float = 0.1, spectral_radius: float = 0.95):
        super(ReservoirLayer, self).__init__()
        
        # Input weights (random, usually fixed)
        self.W_in = torch.rand(reservoir_size, input_dim) * 2 - 1  # Input weights in range [-1, 1]
        
        # Reservoir weights (random, sparse, usually fixed)
        self.W_res = torch.rand(reservoir_size, reservoir_size) * 2 - 1  # Reservoir weights in range [-1, 1]
        
        # Apply sparsity by setting a portion of the weights to zero
        mask = torch.rand(reservoir_size, reservoir_size) > sparsity
        self.W_res = self.W_res * mask.float()

        # Normalize to achieve the spectral radius constraint using torch.linalg.eig
        eigenvalues = torch.linalg.eigvals(self.W_res)  # Compute eigenvalues
        max_eigenvalue = torch.max(torch.abs(eigenvalues))  # Take the largest eigenvalue
        self.W_res *= spectral_radius / max_eigenvalue  # Rescale the reservoir weights to achieve the desired spectral radius

    def forward(self, x):
        # Ensure x has the shape [batch_size, input_dim]
        if x.dim() > 2:  # If there's an extra dimension, squeeze it
            x = x.squeeze(-1)  # Remove the last dimension if it's size 1

        # # Now x should have shape [batch_size, input_dim]
        # print(x.shape)  # Should print [batch_size, input_dim]
        # print(self.W_in.T.shape)  # Should print [input_dim, reservoir_size]
        
        batch_size = x.size(0)
        reservoir_size = self.W_res.size(0)
        
        # Initialize the reservoir state with shape [batch_size, reservoir_size]
        if not hasattr(self, 'state') or self.state.size(0) != batch_size:
            self.state = torch.zeros(batch_size, reservoir_size, device=x.device)

        # Update reservoir state
        # x @ W_in.T has shape [batch_size, reservoir_size]
        # self.state @ W_res has shape [batch_size, reservoir_size]
        self.state = torch.tanh(torch.matmul(x, self.W_in.T) + torch.matmul(self.state, self.W_res))
        
        return self.state  # Output should be [batch_size, reservoir_size]




class EchoStateNetwork(torch.nn.Module):
    def __init__(self, input_dim: int, reservoir_size: int, output_size: int, sparsity: float = 0.1,
                 spectral_radius: float = 0.95, layer_units: List[int] = [128, 64], init_weights: bool = True, 
                 exogenous_dim: int = 0):
        super(EchoStateNetwork, self).__init__()

        # Define the reservoir layer (fixed weights)
        self.reservoir = ReservoirLayer(input_dim, reservoir_size, sparsity, spectral_radius)

        # Define the MLP layers that process the output of the reservoir
        activation = torch.nn.ReLU()

        if len(layer_units) == 1:
            # Directly output to the number of outputs
            layers = [torch.nn.Linear(reservoir_size + exogenous_dim, output_size)]
        else:
            # Input layer
            layers = [torch.nn.Linear(reservoir_size + exogenous_dim, layer_units[0]), activation]
            # Hidden layers
            for i in range(len(layer_units) - 1):
                layers.append(torch.nn.Linear(layer_units[i], layer_units[i + 1]))
                layers.append(activation)
            # Output layer
            layers.append(torch.nn.Linear(layer_units[-1], output_size))

        # Store the MLP as a sequential model
        self.MLP_layers = torch.nn.Sequential(*layers)

        # Initialize weights if requested
        if init_weights:
            self.MLP_layers.apply(self._init_weights)

    def forward(self, x_seq, exogenous_data=None, device="cpu", y_hist=None):
        # Get the batch size and sequence length
        batch_size = x_seq.size(0)
        sequence_length = x_seq.size(1)
        input_dim = x_seq.size(2)

        # Initialize the reservoir states storage
        states = []

        # Loop over the sequence length and collect reservoir states
        for t in range(sequence_length):
            x_t = x_seq[:, t, :]  # Extract the input at time step t (shape [batch_size, input_dim])
            state = self.reservoir(x_t)  # Get the reservoir state for this time step (shape [batch_size, reservoir_size])
            states.append(state)

        # Stack the reservoir states into a single tensor
        states = torch.stack(states, dim=0)  # Now states shape will be [sequence_length, batch_size, reservoir_size]
        # print("Reservoir states shape:", states.shape)  # Should be [sequence_length, batch_size, reservoir_size]

        # Take the last reservoir state for prediction (from the last time step)
        out = states[-1, :, :]  # Select the last time step (shape [batch_size, reservoir_size])
        # print("Last reservoir state shape:", out.shape)  # Should be [batch_size, reservoir_size]

        # Append exogenous data if provided
        if exogenous_data is not None:
            out = torch.cat((out, exogenous_data), dim=1)

        # Pass through the MLP layers
        out = self.MLP_layers(out)
        # print("Output after MLP shape:", out.shape)  # Should be [batch_size, output_size]
        return out





    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()