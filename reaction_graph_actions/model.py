import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter_max, scatter_add, \
                          scatter_mean


class MessageLayer(nn.Module):
    """
    Class defining the message passing operation on the composition graph
    """
    def __init__(self, fea_len, action_fea_len, num_heads=1):
        """
        Inputs
        ----------
        fea_len: int
            Number of precursor hidden features.
        action_len: int
            Number of action encoded features
        """
        super(MessageLayer, self).__init__()

        # Pooling and Output
        hidden_ele = [256]
        hidden_msg = [256]
        self.pooling = nn.ModuleList([WeightedAttention(
            gate_nn=SimpleNetwork(2*fea_len + action_fea_len, 1, hidden_ele),
            message_nn=SimpleNetwork(2*fea_len + action_fea_len, fea_len, hidden_msg),
            # message_nn=nn.Linear(2*fea_len, fea_len),
            # message_nn=nn.Identity(),
            ) for _ in range(num_heads)])

    def forward(self, prec_weights, prec_in_fea,
                self_fea_idx, nbr_fea_idx, reaction_prec_idx, actions):
        """
        Forward pass

        Parameters
        ----------
        N: Total number of precursors (nodes) in the batch
        M: Total number of precursor pairs in the batch
        C: Total number of reactions (graphs) in the batch
        a_fea: Action sequence encoded features

        Inputs
        ----------
        prec_weights: Variable(torch.Tensor) shape (N,)
            The molar amounts of the precursors in the reaction (normalised)
        prec_in_fea: Variable(torch.Tensor) shape (N, prec_fea_len)
            Precursor hidden features before message passing
        self_fea_idx: torch.Tensor shape (M,)
            Indices of the precursor each of the M edges correspond to
        nbr_fea_idx: torch.Tensor shape (M,)
            Indices of the neighbours of the M edges connect to
        reaction_prec_idx: list of torch.LongTensor of length C
            Mapping from the prec idx to reaction idx
        actions: torch.Tensor shape (C, a_fea)

        Returns
        -------
        prec_out_fea: nn.Variable shape (N, prec_fea_len)
            Precursor hidden features after message passing
        """
        # construct the total features for passing
        prec_nbr_weights = prec_weights[nbr_fea_idx, :]
        prec_nbr_fea = prec_in_fea[nbr_fea_idx, :]
        prec_self_fea = prec_in_fea[self_fea_idx, :]
        prec_to_reaction = [reaction_prec_idx[i] for i in self_fea_idx]
        global_state = actions[prec_to_reaction, :]  # action embedding - for each reaction
 
        fea = torch.cat([prec_self_fea, prec_nbr_fea, global_state], dim=1)   #self, neighbours and action_embed

        # sum selectivity (attention) over the neighbours to get elems
        head_fea = []
        for attnhead in self.pooling:
            head_fea.append(attnhead(fea=fea,
                                     index=self_fea_idx,
                                     weights=prec_nbr_weights))

        # average over attention heads
        fea = torch.mean(torch.stack(head_fea), dim=0)

        # update by adding to existing feature
        return fea + prec_in_fea

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)


class ReactionNet(nn.Module):
    """
    Create a neural network for predicting elements in product from precursors.

    The ReactionNet model is comprised of a fully connected output network
    and message-passing graph layers.

    The message-passing layers are used to determine a descriptor set
    for the fully connected network. Critically the graphs are used to
    generate learned representations of reactions from their precursors.

    Uses pretrained rnn for action sequence encoding
    """
    def __init__(self, pretrained_rnn, 
                    orig_prec_fea_len, 
                    prec_fea_len, 
                    n_graph, 
                    intermediate_dim,
                    target_dim,
                    mask):
        """
        Initialize ReactionNet.

        Inputs
        ----------
        pretrained_rnn: torch model
            RNN autoencoder pretrained
        orig_prec_fea_len: int
            Number of precursor features in the input.
        prec_fea_len: int
            Number of hidden precursor features in the graph layers
        n_graph: int
            Number of graph layers
        intermediate_dim: int
            Intermediate layers dimension in output network
        target_dim: int
            Target embedding dimension
        mask: bool
            Whether to mask with precursor elements
        """

        super(ReactionNet, self).__init__()

        self.mask = mask
        self.pretrained_model = pretrained_rnn

        # apply linear transform to the input to get a trainable embedding
        self.embedding = nn.Linear(orig_prec_fea_len, prec_fea_len-1, bias=False)

        # create a list of Message passing layers
        msg_heads = 3   # hard coded
        self.graphs = nn.ModuleList(
                        [MessageLayer(prec_fea_len, pretrained_rnn.latent_dim, msg_heads)
                            for i in range(n_graph)])

        # define a global pooling function for the reaction
        react_heads = 3
        react_hidden = [256]
        msg_hidden = [256]
        self.cry_pool = nn.ModuleList([WeightedAttention(
            gate_nn=SimpleNetwork(prec_fea_len, 1, react_hidden),
            message_nn=SimpleNetwork(prec_fea_len, prec_fea_len, msg_hidden),
            # message_nn=nn.Linear(prec_fea_len, prec_fea_len),
            # message_nn=nn.Identity(),
            ) for _ in range(react_heads)])

        # define an output neural network for element prediction
        #out_hidden = [1024, 512, 256, 256, 128]
        #out_hidden = [intermediate_dim*4, intermediate_dim*2, intermediate_dim*2, intermediate_dim, intermediate_dim]
        out_hidden = [intermediate_dim, intermediate_dim*2, intermediate_dim*2, intermediate_dim]
        self.output_nn = ResidualNetwork(prec_fea_len, target_dim, out_hidden)
        # self.output_nn = SimpleNetwork(elem_fea_len, target_dim, out_hidden)
     
    def forward(self, prec_weights, orig_prec_fea, self_fea_idx,
                nbr_fea_idx, reaction_prec_idx, actions_padded, actions_len, prec_elem_mask):
        """
        Forward pass

        Parameters
        ----------
        N: Total number of precursors (nodes) in the batch
        M: Total number of combinations of precursors (edges) in the batch
        C: Total number of reactions (graphs) in the batch
        A: Max number of actions in batch
        a_fea: Action embedding dimension

        Inputs
        ----------
        orig_prec_fea: Variable(torch.Tensor) shape (N, orig_prec_fea_len)
            Precursor features of each of the N precursors in the batch
        self_fea_idx: torch.Tensor shape (M,)
            Indices of the precursor each of the M edges correspond to
        nbr_fea_idx: torch.Tensor shape (M,)
            Indices of the neighbours of the M edges connect to
        reaction_prec_idx: torch.LongTensor (C,)
            Mapping from the prec idx to reaction idx
        actions_padded: torch.Tensor shape (C, A, a_fea)
            Padded action sequences for input into pretrained RNN
        actions_len: torch.LongTensor shape (C)
            Length of each action sequence per crystal (reaction) for input into RNN

        Returns
        -------
        output: nn.Variable shape (C, target_dim)
            Output elemental probabilities
        react_fea: nn.Variable shape (C, prec_fea_len)
            learned reaction representation
        """

        # get fixed length representations of the action sequences for each reaction
        _, _, actions = self.pretrained_model(actions_padded, actions_len)

        # embed the original features into the graph layer description
        prec_fea = self.embedding(orig_prec_fea)

        # do this so that we can examine the embeddings without
        # influence of the weights
        prec_fea = torch.cat([prec_fea, prec_weights], dim=1)

        # apply the graph message passing functions
        for graph_func in self.graphs:
            prec_fea = graph_func(prec_weights, prec_fea,
                                  self_fea_idx, nbr_fea_idx, reaction_prec_idx, actions)


        # generate reaction features by pooling the precursor features
        head_fea = []
        for attnhead in self.cry_pool:
            head_fea.append(attnhead(fea=prec_fea,
                                     index=reaction_prec_idx,
                                     weights=prec_weights))

        # get a learned representation of the reaction by averaging
        react_fea = torch.mean(torch.stack(head_fea), dim=0)

        # apply neural network to map from learned features to target
        output = self.output_nn(react_fea)

        if self.mask:
            # mask the output with the prec elems
            zero_prob = torch.zeros_like(prec_elem_mask)
            zero_prob[zero_prob == 0] = -999
            output = torch.where(prec_elem_mask != 0, output, zero_prob)

        # output AND the reaction representation
        return output, react_fea

    """def __repr__(self):
        return '{}'.format(self.__class__.__name__)"""


class WeightedMeanPooling(torch.nn.Module):
    """
    mean pooling
    """
    def __init__(self):
        super(WeightedMeanPooling, self).__init__()

    def forward(self, fea, index, weights):
        fea = weights * fea
        return scatter_mean(fea, index, dim=0)

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)


class WeightedAttention(nn.Module):
    """
    Weighted softmax attention layer
    """
    def __init__(self, gate_nn, message_nn, num_heads=1):
        """
        Inputs
        ----------
        gate_nn: Variable(nn.Module)
        """
        super(WeightedAttention, self).__init__()
        self.gate_nn = gate_nn
        self.message_nn = message_nn

    def forward(self, fea, index, weights):
        """ forward pass """

        gate = self.gate_nn(fea)
        # gate is the attention coeffs (for pairs)
        gate = gate - scatter_max(gate, index, dim=0)[0][index]
        gate = weights * gate.exp()
        gate = gate / (scatter_add(gate, index, dim=0)[index] + 1e-13)

        fea = self.message_nn(fea)
        out = scatter_add(gate * fea, index, dim=0)

        return out

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