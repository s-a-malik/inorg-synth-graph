from tqdm.autonotebook import trange

import torch
import torch.nn as nn
from torch.nn.functional import l1_loss as mae
from torch.nn.functional import mse_loss as mse
from torch_scatter import scatter_max, scatter_add

from matgps.utils import AverageMeter
from matgps.segments import SimpleNetwork, ResidualNetwork


class StoichNet(nn.Module):
    """
    Create a neural network for predicting material stoichiometry from elements
    present and context vector

    """
    def __init__(
        self,
        orig_elem_fea_len,
        orig_reaction_fea_len,
        intermediate_dim,
        n_heads
    ):
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

        embed_dims = [
            intermediate_dim,
            intermediate_dim,
            intermediate_dim,
            intermediate_dim,
            128,
            128,
            64
        ]

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
            stoich = attnhead(fea=elem_fea_with_reaction, index=reaction_elem_idx)
            head_stoich.append(stoich)

        stoichs = torch.mean(torch.stack(head_stoich), dim=0)
        stoichs = stoichs.view(-1)

        return stoichs

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)

    def evaluate(self, generator, criterion, optimizer, device, task="train", verbose=False):
        """
        evaluate the model
        """

        if task == "test":
            self.eval()
            test_targets = []
            test_pred = []
            test_ids = []
            test_comp = []
            test_total = 0
            test_crys_ids = []
        else:
            loss_meter = AverageMeter()
            rmse_meter = AverageMeter()
            mae_meter = AverageMeter()
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

                # compute output
                output = self(*input_)

                if task == "test":
                    # collect the model outputs
                    test_ids += batch_ids
                    test_comp += batch_comp
                    test_targets += target.tolist()
                    test_pred += output.tolist()
                    test_total += len(batch_ids)
                else:
                    # get predictions and error
                    loss = criterion(output, target)
                    loss_meter.update(loss.data.cpu().item(), target.size(0))

                    mae_error = mae(output, target)
                    mae_meter.update(mae_error, target.size(0))

                    rmse_error = mse(output, target).sqrt_()
                    rmse_meter.update(rmse_error, target.size(0))
                    if task == "train":
                        # compute gradient and do SGD step
                        optimizer.zero_grad()
                        loss.backward()
                        # plot_grad_flow(model.named_parameters())
                        optimizer.step()
                t.update()

        if task == "test":
            return test_ids, test_comp, test_pred, test_targets, test_total
        else:
            return loss_meter.avg, mae_meter.avg, rmse_meter.avg


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


if __name__ == "__main__":
    pass