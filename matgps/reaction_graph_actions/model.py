import numpy as np

from tqdm.autonotebook import trange

import torch
import torch.nn as nn

from matgps.utils import AverageMeter
from matgps.segments import WeightedAttention, SimpleNetwork, ResidualNetwork


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
    def __init__(
        self,
        pretrained_rnn,
        orig_prec_fea_len,
        prec_fea_len,
        n_graph,
        intermediate_dim,
        target_dim,
        mask
    ):
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
        # out_hidden = [1024, 512, 256, 256, 128]
        # out_hidden = [intermediate_dim*4, intermediate_dim*2, intermediate_dim*2, intermediate_dim, intermediate_dim]
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

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)

    def evaluate(self, generator, criterion, optimizer, device, threshold, task="train", verbose=False):
        """
        evaluate the model
        """

        if task == "test":
            self.eval()
            test_targets = []
            test_pred = []
            test_react_embed = []
            test_ids = []
            test_comp = []
            test_total = 0
            subset_accuracy = 0
        else:
            loss_meter = AverageMeter()
            if task == "val":
                self.eval()
            elif task == "train":
                self.train()
            else:
                raise NameError("Only train, val or test is allowed as task")

        with trange(len(generator), disable=(not verbose)) as t:
            for input_, target, batch_comp, batch_ids in generator:

                # move tensors to GPU
                input_ = (tensor.to(device) for tensor in input_)
                target = target.to(device)
                #print(target)

                # compute output
                output, react_embed = self(*input_)

                if task == "test":

                    # collect the model outputs
                    test_ids += batch_ids
                    test_comp += batch_comp
                    test_targets += target.tolist()
                    test_pred += output.tolist()
                    test_react_embed += react_embed.tolist()
                    # add threshold and get element prediction
                    logit_threshold = torch.tensor(threshold/ (1 - threshold)).log()
                    test_elems = output > logit_threshold   # bool 2d array
                    target_elems = target != 0                # bool array

                    # metrics:
                    # fully correct - subset accuracy
                    correct_row = [torch.all(test_elems[x].eq(target_elems[x])) for x in range(len(test_elems))]
                    subset_accuracy += np.count_nonzero(correct_row)   # number of perfect matches in batch
                    test_total += target.size(0)
                else:
                    # get predictions and error
                    # make targets into labels for classification
                    target_labels = torch.where(target != 0, torch.ones_like(target), target)
                    loss = criterion(output, target_labels)
                    loss_meter.update(loss.data.cpu().item(), target.size(0))

                    if task == "train":
                        # compute gradient and do SGD step
                        optimizer.zero_grad()
                        loss.backward()
                        # plot_grad_flow(self.named_parameters())
                        optimizer.step()

                t.update()

        if task == "test":
            return (
                test_ids,
                test_comp,
                test_pred,
                test_react_embed,
                test_targets,
                subset_accuracy,
                test_total
            )
        else:
            return loss_meter.avg


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


if __name__ == "__main__":
    pass