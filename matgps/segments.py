import torch.nn as nn

from torch_scatter import scatter_max, scatter_add, \
                          scatter_mean


class WeightedMeanPooling(nn.Module):
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