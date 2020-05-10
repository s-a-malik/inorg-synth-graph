import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NoGraphNet(nn.Module):
    """
    Simple Residal network with stoich/magpie inputs, output predicted elements in target
    """
    def __init__(self, max_prec,
                    embedding_dim,
                    intermediate_dim,
                    target_dim,
                    mask):
        """
        Initialize NoGraphNet.

        Inputs
        ----------
        max_prec: int
            Number of precursors used
        embedding_dim: int
            Number of precursor input features
        intermediate_dim: int
            Intermediate layers dimension in output nn
        target_dim: int
            Target embedding dimension
        mask: bool
            Whether to mask with precursor elements
        """
        super(NoGraphNet, self).__init__()

        self.mask = mask
        # define an output neural network for element prediction
        #out_hidden = [1024, 512, 256, 256, 128]
        #out_hidden = [intermediate_dim*4, intermediate_dim*2, intermediate_dim*2, intermediate_dim, intermediate_dim]
        out_hidden = [intermediate_dim, intermediate_dim*2, intermediate_dim*2, intermediate_dim]
        self.output_nn = ResidualNetwork(max_prec*embedding_dim, target_dim, out_hidden)

     
    def forward(self, precs, prec_elem_mask):
        """
        Forward pass

        Parameters
        ----------
        C: Total number of reactions (graphs) in the batch
        max_prec: max number of precursors in a reaction
        embedding_dim: Input precursor embedding dimension
        target_dim: Number of elements in dataset, the dimension of the target composition vector

        Inputs
        ----------
        input: (C, max_prec*embedding_dim)
        prec_elem_mask: (C, target_dim)   

        Returns
        -------
        output: nn.Variable shape (C, target_dim)
            Output elemental probabilities
        """
        # reshape to concat the precursors
        precs = precs.view(precs.shape[0], -1)
        output = self.output_nn(precs)

        if self.mask:
            # mask the output with the prec elems
            zero_prob = torch.zeros_like(prec_elem_mask)
            zero_prob[zero_prob == 0] = -999
            output = torch.where(prec_elem_mask != 0, output, zero_prob)

        # output AND the reaction representation (in this case this is the input)
        return output, precs


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
        # self.bns = nn.ModuleList([nn.BatchNorm1d(dims[i+1])
        #                           for i in range(len(dims)-1)])
        self.res_fcs = nn.ModuleList([nn.Linear(dims[i], dims[i+1], bias=False)
                                      if (dims[i] != dims[i+1])
                                      else nn.Identity()
                                      for i in range(len(dims)-1)])
        self.acts = nn.ModuleList([nn.ReLU() for _ in range(len(dims)-1)])

        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, fea):
        # for fc, bn, res_fc, act in zip(self.fcs, self.bns,
        #                                self.res_fcs, self.acts):
        #     fea = act(bn(fc(fea)))+res_fc(fea)
        for fc, res_fc, act in zip(self.fcs, self.res_fcs, self.acts):
            fea = act(fc(fea))+res_fc(fea)

        return self.fc_out(fea)

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)


class SecondNet(nn.Module):
    """Net for transer with added trainable layers to learn stoichiometry
    """

    def __init__(self, pretrained_model, threshold):
        """
        Input:
        pretrained_model:
            CompositionNet pretrained model which outputs logits for elemental presence
        threshold: float
            hyperparameter for probabilistic threshold for choosing which elements present
        """
        super(SecondNet, self).__init__()
        self.pretrained_model = pretrained_model    # load in this model for transfer

        # threshold:
        threshold = np.log(threshold/(1-threshold)) # convert to logit threshold
        self.threshold = nn.Threshold(threshold, 0)

        out_hidden = [256, 256, 128]
        # input dimension is elem_fea_len (i.e. what it was before training for elements) + target_dim
        # output dimension is target_dim
        self.last_nn = SimpleNetwork(pretrained_model.output_nn.fcs[0].in_features + pretrained_model.output_nn.fc_out.out_features, 
                                    pretrained_model.output_nn.fc_out.out_features,
                                    out_hidden)


    def forward(self, elem_weights, orig_elem_fea, self_fea_idx,
                nbr_fea_idx, crystal_elem_idx):
        
        output, crys_fea = self.pretrained_model(elem_weights, orig_elem_fea, self_fea_idx,
                nbr_fea_idx, crystal_elem_idx)
        # threshold
        output = self.threshold(output)
        
        #concatenate with precursor mixture embedding
        output_with_context = torch.cat([output, crys_fea], dim=1)

        stoich = self.last_nn(output_with_context)

        return F.relu(stoich), output



if __name__ == "__main__":
    pass