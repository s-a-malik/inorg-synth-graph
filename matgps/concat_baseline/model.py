import numpy as np

from tqdm.autonotebook import trange

import torch
import torch.nn as nn

from matgps.utils import AverageMeter
from matgps.segments import ResidualNetwork


class ConcatNet(nn.Module):
    """
    Simple Residal network with stoich/magpie inputs, output predicted elements in target
    """
    def __init__(
        self,
        max_prec,
        embedding_dim,
        intermediate_dim,
        target_dim,
        mask
    ):
        """
        Initialize ConcatNet.

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
        super(ConcatNet, self).__init__()

        self.mask = mask
        # define an output neural network for element prediction
        # out_hidden = [1024, 512, 256, 256, 128]
        # out_hidden = [intermediate_dim*4, intermediate_dim*2, intermediate_dim*2, intermediate_dim, intermediate_dim]
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
        # NOTE using just the concatenated magpie representations could be improved upon.
        return output, precs

    def evaluate(self, generator, criterion, optimizer, device, threshold, task="train", verbose=False):
        """
        evaluate the model
        """

        if task == "test":
            self.eval()
            test_targets = []
            test_pred = []
            test_prec_embed = []
            test_ids = []
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
            for input_, target, batch_ids in generator:

                # move tensors to GPU
                input_ = (tensor.to(device) for tensor in input_)
                target = target.to(device)
                # print(target)

                # compute output
                output, prec_embed = self(*input_)
                # print("output", output)

                if task == "test":

                    # collect the model outputs
                    test_ids += batch_ids
                    test_targets += target.tolist()
                    test_pred += output.tolist()
                    test_prec_embed += prec_embed.tolist()
                    # add threshold and get element prediction
                    logit_threshold = torch.tensor(threshold/ (1 - threshold)).log()
                    test_elems = output > logit_threshold   # bool 2d array
                    target_elems = target != 0              # bool array
                    # print(np.shape(test_elems))
                    # print(np.shape(target_elems))

                    # metrics:
                    # fully correct - subset accuracy
                    correct_row = [torch.all(test_elems[x].eq(target_elems[x])) for x in range(len(test_elems))]
                    # print(np.shape(correct_row))
                    subset_accuracy += np.count_nonzero(correct_row)   # number of perfect matches in batch
                    # print(subset_accuracy)
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
                test_pred,
                test_prec_embed,
                test_targets,
                subset_accuracy,
                test_total
            )
        else:
            return loss_meter.avg





if __name__ == "__main__":
    pass