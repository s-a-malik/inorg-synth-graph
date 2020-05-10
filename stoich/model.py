import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter_max, scatter_add, \
                          scatter_mean


class StoichNet(nn.Module):
    """
    Create a neural network for predicting material stoichiometry from elements 
    present and context vector
    
    """
    def __init__(self, orig_elem_fea_len, 
                    orig_reaction_fea_len,
                    intermediate_dim,
                    n_heads):
        """
        Initialize StoichNet.

        Inputs
        ----------
        orig_elem_fea_len: int
            Number of elem features in the input.
        orig_react_fea_len: int
            Number of hidden reaction representation features
        n_heads: int
            Number of heads to average over
        """
        super(StoichNet, self).__init__()
        
        embed_dims = [intermediate_dim, intermediate_dim, intermediate_dim, intermediate_dim, 128, 128, 64]
    
        # use hyperparameter for number of attn heads
        self.stoich_pool = nn.ModuleList([NormStoich(
            gate_nn=ResidualNetwork(orig_elem_fea_len+orig_reaction_fea_len, 1, embed_dims)
            ) for _ in range(n_heads)])


    def forward(self, orig_elem_fea, reaction_elem_idx, reaction_embed):
        """
        Forward pass

        Parameters
        ----------
        N: Total number of elems in the batch
        C: Total number of reactions in the batch

        Inputs
        ----------
        orig_elem_fea: torch.Tensor shape (N, elem_fea_len)
            Atom features of each of the N elems in the batch
        reaction_elem_idx: torch.LongTensor (N,)
            Mapping from the elem idx to reaction idx
        reaction_embed: torch.Tensor (C,)
            Mapping from reaction idx to reaction_embedding for that reaction

        Returns
        -------
        stoichs: Variable(torch.Tensor) shape (N,)
            the normalised stoichiometries of the elements
        """  

        # embed the original features into the graph layer description
        reaction_embed_per_elem = reaction_embed[reaction_elem_idx, :]
        elem_fea_with_reaction = torch.cat([orig_elem_fea, reaction_embed_per_elem], dim=1)
        
        # without prec embed
        # elem_fea = self.embedding(orig_elem_fea)

        head_stoich = []
        for attnhead in self.stoich_pool:
            stoich = attnhead(fea=elem_fea_with_reaction,
                                     index=reaction_elem_idx)
            head_stoich.append(stoich)

        stoichs = torch.mean(torch.stack(head_stoich), dim=0)
        stoichs = stoichs.view(-1)

        return stoichs

    """def __repr__(self):
        return '{}'.format(self.__class__.__name__)"""

class NormStoich(nn.Module):
    """
    Softmax layer to normalise stoichiometries w.r.t. reactions
    """
    def __init__(self, gate_nn):
        """
        Inputs
        ----------
        gate_nn: Variable(nn.Module)
        """
        super(NormStoich, self).__init__()

        self.gate_nn = gate_nn

    def forward(self, fea, index):
        """ forward pass. Returns normalised stoichiometries """    

        gate = self.gate_nn(fea)
        gate = gate - scatter_max(gate, index, dim=0)[0][index]
        gate = gate.exp()
        gate = gate / (scatter_add(gate, index, dim=0)[index] + 1e-13)

        return gate

    def __repr__(self):
        return '{}(gate_nn={})'.format(self.__class__.__name__,
                                       self.gate_nn)


class SimpleNetwork(nn.Module):
    """
    Simple Feed Forward Neural Network
    """
    def __init__(self, input_dim, output_dim, hidden_layer_dims):
        """
        Inputs
        ----------
        input_dim: int
        output_dim: int
        hidden_layer_dims: list(int)
        """
        super(SimpleNetwork, self).__init__()

        dims = [input_dim]+hidden_layer_dims

        self.fcs = nn.ModuleList([nn.Linear(dims[i], dims[i+1])
                                  for i in range(len(dims)-1)])
        self.acts = nn.ModuleList([nn.LeakyReLU() for _ in range(len(dims)-1)])

        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, fea):
        for fc, act in zip(self.fcs, self.acts):
            fea = act(fc(fea))

        return self.fc_out(fea)

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)

class ResidualNetwork(nn.Module):
    """
    Feed forward Residual Neural Network
    """
    def __init__(self, input_dim, output_dim, hidden_layer_dims):
        """
        Inputs
        ----------
        input_dim: int
        output_dim: int
        hidden_layer_dims: list(int)

        """
        super(ResidualNetwork, self).__init__()

        dims = [input_dim]+hidden_layer_dims

        self.fcs = nn.ModuleList([nn.Linear(dims[i], dims[i+1])
                                  for i in range(len(dims)-1)])
        self.res_fcs = nn.ModuleList([nn.Linear(dims[i], dims[i+1], bias=False)
                                      if (dims[i] != dims[i+1])
                                      else nn.Identity()
                                      for i in range(len(dims)-1)])
        self.acts = nn.ModuleList([nn.ReLU() for _ in range(len(dims)-1)])

        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, fea):

        for fc, res_fc, act in zip(self.fcs, self.res_fcs, self.acts):
            fea = act(fc(fea))+res_fc(fea)

        return self.fc_out(fea)

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)


if __name__ == "__main__":
    pass