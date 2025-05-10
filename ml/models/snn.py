# imports
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen
from typing import List
from snntorch import surrogate

import snntorch
import torch
import torch.nn as nn

# Leaky
class SNN(nn.Module):
    """Simple spiking neural network in snntorch."""
    def __init__(self, input_dim, timesteps, hidden, exogenous_dim, out_dim):
        super().__init__()
        torch.manual_seed(0)
        self.timesteps = timesteps # number of time steps to simulate the network
        self.hidden = hidden # number of hidden neurons
        spike_grad = surrogate.fast_sigmoid() # surrogate gradient function

        print("----------------------------------------------------", input_dim)
        print("----------------------------------------------------", out_dim)
        

        # randomly initialize decay rate and threshold for layer 1
        beta_in = torch.rand(self.hidden)
        thr_in = torch.rand(self.hidden)

        # layer 1
        self.fc_in = torch.nn.Linear(in_features=input_dim, out_features=self.hidden)
        self.lif_in = snn.Leaky(beta=beta_in, threshold=thr_in, learn_beta=True, spike_grad=spike_grad)

        # randomly initialize decay rate and threshold for layer 2
        beta_hidden = torch.rand(self.hidden)
        thr_hidden = torch.rand(self.hidden)


        # layer 2
        self.fc_hidden = torch.nn.Linear(in_features=self.hidden, out_features=self.hidden)
        self.lif_hidden = snn.Leaky(beta=beta_hidden, threshold=thr_hidden, learn_beta=True, spike_grad=spike_grad)

        # randomly initialize decay rate for output neuron
        beta_out = torch.rand(1)

        # layer 3: leaky integrator neuron. Note the reset mechanism is disabled and we will disregard output spikes.
        self.fc_out = torch.nn.Linear(in_features=self.hidden, out_features=out_dim)
        self.li_out = snn.Leaky(beta=beta_out, threshold=1.0, learn_beta=True, spike_grad=spike_grad, reset_mechanism="none")
    def forward(self, x, exogenous_data=None, device='cpu', y_hist=None):
        """Forward pass for several time steps."""
        torch.manual_seed(0)
        # Initialize membrane potential
        mem_1 = self.lif_in.init_leaky()
        mem_2 = self.lif_hidden.init_leaky()
        mem_3 = self.li_out.init_leaky()

        # Loop over timesteps
        for step in range(self.timesteps):
            x_timestep = x[:, :, :, 0] 

            print('\n\n\n\n\n\ntimestep', x_timestep.shape)
            # print(x.shape) 
            # Reshape x_timestep to match the expected input shape
            # x_timestep = x_timestep.view(x_timestep.size(0), -1)
            x_timestep = x_timestep[0]
            # print('timestep after view', x_timestep.shape)
            print("x_timestep.shape\n\n\n\n\n---------------------------------------------------------------------------------------------\n\n\n\n\n", x_timestep.shape)
            cur_in = self.fc_in(x_timestep)

            spk_in, mem_1 = self.lif_in(cur_in, mem_1)

            # print('mem_1', mem_1.shape)
            cur_hidden = self.fc_hidden(spk_in)
            spk_hidden, mem_2 = self.lif_hidden(cur_hidden, mem_2)

            # print('mem_2', mem_2.shape)
            cur_out = self.fc_out(spk_hidden)
            _, mem_3 = self.li_out(cur_out, mem_3)

            # print('mem_3', mem_3.shape)
            # break

        # Stack recorded membrane potentials
        # output_tensor = mem_3.squeeze(1)
        output_tensor = mem_3.mean(dim=0)
    
        print("mem_3", mem_3.shape)
        # print("output tensor", output_tensor.shape)
        output_tensor = torch.tensor([[output_tensor]])
        # print('output', output_tensor.shape)
        return output_tensor


# ----------------------------------------------------------
    # With Lapicque
# class SNN(nn.Module):
#     """Simple spiking neural network in snntorch."""
#     def __init__(self, input_dim, timesteps, hidden, exogenous_dim, out_dim):
#         super().__init__()
#         torch.manual_seed(0)
#         self.timesteps = timesteps # number of time steps to simulate the network
#         self.hidden = hidden # number of hidden neurons
#         spike_grad = surrogate.fast_sigmoid() # surrogate gradient function

#         # randomly initialize decay rate and threshold for layer 1

#         thr_in = torch.rand(self.hidden)

        
#         # layer 1
#         self.fc_in = torch.nn.Linear(in_features=input_dim, out_features=self.hidden)
#         self.lif_in = snn.Lapicque(R=5, C=1e-4, threshold=thr_in, spike_grad=spike_grad)

#         # randomly initialize decay rate and threshold for layer 2
#         beta_hidden = torch.rand(self.hidden)
#         thr_hidden = torch.rand(self.hidden)


#         # layer 2
#         self.fc_hidden = torch.nn.Linear(in_features=self.hidden, out_features=self.hidden)
#         self.lif_hidden = snn.Leaky(beta=beta_hidden, threshold=thr_hidden, learn_beta=True, spike_grad=spike_grad)
#         print(self.lif_hidden)
#         # randomly initialize decay rate for output neuron
#         beta_out = torch.rand(1)

#         # layer 3: leaky integrator neuron. Note the reset mechanism is disabled and we will disregard output spikes.
#         self.fc_out = torch.nn.Linear(in_features=self.hidden, out_features=out_dim)
#         self.li_out = snn.Leaky(beta=beta_out, threshold=1.0, learn_beta=True, spike_grad=spike_grad, reset_mechanism="none")
#     def forward(self, x, exogenous_data=None, device='cuda', y_hist=None):
#         """Forward pass for several time steps."""
#         torch.manual_seed(0)
#         # Initialize membrane potential
#         mem_1 = self.lif_in.init_leaky()
#         mem_2 = self.lif_hidden.init_leaky()
#         mem_3 = self.li_out.init_leaky()

#         # Loop over timesteps
#         for step in range(self.timesteps):
#             x_timestep = x[:, :, :, 0] 

#             # print('timestep', x_timestep.shape)
#             # print(x.shape) 
#             # Reshape x_timestep to match the expected input shape
#             x_timestep = x_timestep.view(x_timestep.size(0), -1)
           
#             # print('timestep after view', x_timestep.shape)
#             cur_in = self.fc_in(x_timestep)

            
#             spk_in, mem_1 = self.lif_in(cur_in, mem_1)
            
#             # print('mem_1', mem_1.shape)
#             cur_hidden = self.fc_hidden(spk_in)
            
#             spk_hidden, mem_2 = self.lif_hidden(cur_hidden, mem_2)
            
#             # print('mem_2', mem_2.shape)
#             cur_out = self.fc_out(spk_hidden)
            
        
#             _, mem_3 = self.li_out(cur_out, mem_3)
            
#             # print('mem_3', mem_3.shape)
#             # break

#         # Stack recorded membrane potentials
#         output_tensor = mem_3.squeeze(1)

#         # print('output', output_tensor.shape)

#         return output_tensor


# -----------------------------------------------------------
# RLeaky

# class SNN(nn.Module):
#     """Simple spiking neural network in snntorch."""
#     def __init__(self, input_dim, timesteps, hidden, exogenous_dim, out_dim):
#         super().__init__()
#         # Reproducibility
#         torch.manual_seed(0)
        
#         self.timesteps = timesteps # number of time steps to simulate the network
#         self.hidden = hidden # number of hidden neurons
#         spike_grad = surrogate.fast_sigmoid() # surrogate gradient function

#         # randomly initialize decay rate and threshold for layer 1
#         beta_in = torch.rand(self.hidden)
#         thr_in = torch.rand(self.hidden)

#         # layer 1
#         self.fc_in = torch.nn.Linear(in_features=input_dim, out_features=self.hidden)
#         self.lif_in  = snn.RLeaky(beta=beta_in, threshold=thr_in, linear_features=self.hidden)


#         # randomly initialize decay rate and threshold for layer 2
#         beta_hidden = torch.rand(self.hidden)
#         thr_hidden = torch.rand(self.hidden)


#         # layer 2
#         self.fc_hidden = torch.nn.Linear(in_features=self.hidden, out_features=self.hidden)
#         self.lif_hidden = snn.RLeaky(beta=beta_hidden, threshold=thr_hidden, linear_features=self.hidden)

#         # randomly initialize decay rate for output neuron
#         beta_out = torch.rand(1)
#         print('bring this beta out', beta_out)
#         # layer 3: leaky integrator neuron. Note the reset mechanism is disabled and we will disregard output spikes.
#         self.fc_out = torch.nn.Linear(in_features=self.hidden, out_features=out_dim)
#         self.li_out = snn.RLeaky(beta=beta_out, threshold=1.0, linear_features=out_dim)
#     def forward(self, x, exogenous_data=None, device='cpu', y_hist=None):
#         torch.manual_seed(0)
#         """Forward pass for several time steps."""
#         # print('this is forwarddd', x.shape)
#         # No need to reshape x
#         # x = x.view(x.size(0), x.size(3), x.size(1), x.size(2))
#         # print('this is forwarddd', x.shape)
#         # Initialize membrane potential
#         spk_in, mem_1 = self.lif_in.init_rleaky()
#         spk_hidden, mem_2 = self.lif_hidden.init_rleaky()
#         _, mem_3 = self.li_out.init_rleaky()

#         # Loop over timesteps
#         for step in range(self.timesteps):
#             x_timestep = x[:, :, :, 0] 

#             # print('timestep', x_timestep.shape)
#             # print(x.shape) 
#             # Reshape x_timestep to match the expected input shape
#             x_timestep = x_timestep.view(x_timestep.size(0), -1)

#             # print('timestep after view', x_timestep.shape)
#             cur_in = self.fc_in(x_timestep)
#             spk_in, mem_1 = self.lif_in(cur_in, spk_in, mem_1)

#             # print('mem_1', mem_1.shape)
#             cur_hidden = self.fc_hidden(spk_in)
#             spk_hidden, mem_2 = self.lif_hidden(cur_hidden, spk_hidden, mem_2)

#             # print('mem_2', mem_2.shape)
#             cur_out = self.fc_out(spk_hidden)
#             _, mem_3 = self.li_out(cur_out, _, mem_3)


#         # Stack recorded membrane potentials
#         output_tensor = mem_3.squeeze(1)
#         # output_tensor = torch.tensor([[output_tensor]])
#         # print('output', output_tensor.shape)
#         return output_tensor


# ---------------------------------------------------------------------

# Synaptic Leakyy

# class SNN(nn.Module):
#     """Simple spiking neural network in snntorch."""
#     def __init__(self, input_dim, timesteps, hidden, exogenous_dim, out_dim):
#         super().__init__()
#         torch.manual_seed(0)
#         self.timesteps = timesteps # number of time steps to simulate the network
#         self.hidden = hidden # number of hidden neurons
#         spike_grad = surrogate.fast_sigmoid() # surrogate gradient function

#         # randomly initialize decay rate and threshold for layer 1
#         alpha_in = torch.rand(self.hidden)
#         beta_in = torch.rand(self.hidden)
#         thr_in = torch.rand(self.hidden)

#         # layer 1
#         self.fc_in = torch.nn.Linear(in_features=input_dim, out_features=self.hidden)
#         self.lif_in = snn.Synaptic(alpha=alpha_in, beta=beta_in, threshold=thr_in, learn_beta=True, spike_grad=spike_grad)

#         # randomly initialize decay rate and threshold for layer 2
#         alpha_hidden = torch.rand(self.hidden)
#         beta_hidden = torch.rand(self.hidden)
#         thr_hidden = torch.rand(self.hidden)


#         # layer 2
#         self.fc_hidden = torch.nn.Linear(in_features=self.hidden, out_features=self.hidden)
#         self.lif_hidden = snn.Synaptic(alpha=alpha_hidden, beta=beta_hidden, threshold=thr_hidden, learn_beta=True, spike_grad=spike_grad)

#         # randomly initialize decay rate for output neuron
#         alpha_out = torch.rand(1)
#         beta_out = torch.rand(1)

#         # layer 3: leaky integrator neuron. Note the reset mechanism is disabled and we will disregard output spikes.
#         self.fc_out = torch.nn.Linear(in_features=self.hidden, out_features=out_dim)
#         self.li_out = snn.Synaptic(alpha=alpha_out, beta=beta_out, threshold=1.0, learn_beta=True, spike_grad=spike_grad, reset_mechanism="none")
#     def forward(self, x, exogenous_data=None, device='cpu', y_hist=None):
#         """Forward pass for several time steps."""
#         torch.manual_seed(0)

#         # Initialize membrane potential
#         syn_1, mem_1 = self.lif_in.init_synaptic()
#         syn_2, mem_2 = self.lif_hidden.init_synaptic()
#         syn_3, mem_3 = self.li_out.init_synaptic()

#         # Loop over timesteps
#         for step in range(self.timesteps):
#             x_timestep = x[:, :, :, 0] 

#             # print('timestep', x_timestep.shape)
#             # print(x.shape) 
#             # Reshape x_timestep to match the expected input shape
#             x_timestep = x_timestep.view(x_timestep.size(0), -1)

#             # print('timestep after view', x_timestep.shape)
#             cur_in = self.fc_in(x_timestep)
#             spk_in, syn_1, mem_1 = self.lif_in(cur_in, syn_1, mem_1)

#             # print('mem_1', mem_1.shape)
#             cur_hidden = self.fc_hidden(spk_in)
#             spk_hidden, syn_2, mem_2 = self.lif_hidden(cur_hidden, syn_2, mem_2)

#             # print('mem_2', mem_2.shape)
#             cur_out = self.fc_out(spk_hidden)
#             _, syn_3, mem_3 = self.li_out(cur_out, syn_3, mem_3)

#             # print('mem_3', mem_3.shape)
#             # break

#         # Stack recorded membrane potentials
#         output_tensor = mem_3.squeeze(1)
#         # output_tensor = torch.tensor([[output_tensor]])
#         # print('output', output_tensor.shape)
#         return output_tensor

# ------------------------------------------------------------------------------------

# Snn Alpha


# class SNN(nn.Module):
#     """Simple spiking neural network in snntorch."""
#     def __init__(self, input_dim, timesteps, hidden, exogenous_dim, out_dim):
#         super().__init__()
#         torch.manual_seed(2)
#         self.timesteps = timesteps # number of time steps to simulate the network
#         self.hidden = hidden # number of hidden neurons
#         spike_grad = surrogate.fast_sigmoid() # surrogate gradient function

#         # randomly initialize decay rate and threshold for layer 1
#         alpha_in = torch.full((self.hidden,), 0.8)
#         beta_in = torch.full((self.hidden,), 0.7)
#         thr_in = torch.rand(self.hidden)

#         # layer 1
#         self.fc_in = torch.nn.Linear(in_features=input_dim, out_features=self.hidden)
#         self.lif_in = snn.Alpha(alpha=alpha_in, beta=beta_in, threshold=thr_in, learn_beta=True, spike_grad=spike_grad)

#         # randomly initialize decay rate and threshold for layer 2
#         alpha_hidden = torch.full((self.hidden,), 0.9)
#         beta_hidden = torch.full((self.hidden,), 0.6)
#         thr_hidden = torch.rand(self.hidden)


#         # layer 2
#         self.fc_hidden = torch.nn.Linear(in_features=self.hidden, out_features=self.hidden)
#         self.lif_hidden = snn.Alpha(alpha=alpha_hidden, beta=beta_hidden, threshold=thr_hidden, learn_beta=True, spike_grad=spike_grad)

#         # randomly initialize decay rate for output neuron
#         alpha_out = torch.full((1,), 0.8)
#         beta_out = torch.full((1,), 0.7)

#         # layer 3: leaky integrator neuron. Note the reset mechanism is disabled and we will disregard output spikes.
#         self.fc_out = torch.nn.Linear(in_features=self.hidden, out_features=out_dim)
#         self.li_out = snn.Alpha(alpha=alpha_out, beta=beta_out, threshold=1.0, learn_beta=True, spike_grad=spike_grad, reset_mechanism="none")
#     def forward(self, x, exogenous_data=None, device='cpu', y_hist=None):
#         """Forward pass for several time steps."""
#         torch.manual_seed(0)
#         # Initialize membrane potential
#         syn_exc_1, syn_inh_1, mem_1 = self.lif_in.init_alpha()
#         syn_exc_2, syn_inh_2, mem_2 = self.lif_hidden.init_alpha()
#         syn_exc_3, syn_inh_3, mem_3 = self.li_out.init_alpha()

#         # Loop over timesteps
#         for step in range(self.timesteps):
#             x_timestep = x[:, :, :, 0] 

#             # print('timestep', x_timestep.shape)
#             # print(x.shape) 
#             # Reshape x_timestep to match the expected input shape
#             x_timestep = x_timestep.view(x_timestep.size(0), -1)

#             # print('timestep after view', x_timestep.shape)
#             cur_in = self.fc_in(x_timestep)
#             spk_in, syn_exc_1, syn_inh_1, mem_1 = self.lif_in(cur_in, syn_exc_1, syn_inh_1, mem_1)

#             # print('mem_1', mem_1.shape)
#             cur_hidden = self.fc_hidden(spk_in)
#             spk_hidden, syn_exc_2, syn_inh_2, mem_2 = self.lif_hidden(cur_hidden, syn_exc_2, syn_inh_2, mem_2)

#             # print('mem_2', mem_2.shape)
#             cur_out = self.fc_out(spk_hidden)
#             _, syn_exc_3, syn_inh_3, mem_3 = self.li_out(cur_out, syn_exc_3, syn_inh_3, mem_3)

#             # print('mem_3', mem_3.shape)
#             # break

#         # Stack recorded membrane potentials
#         output_tensor = mem_3.squeeze(1)
#         # output_tensor = torch.tensor([[output_tensor]])
#         # print('output', output_tensor.shape)
#         return output_tensor
