import os
import torch
from tqdm.autonotebook import trange
import math
import numpy as np

import copy

from tqdm.autonotebook import tqdm
from torch.optim.lr_scheduler import _LRScheduler

from ..utils import AverageMeter

def evaluate(generator, model, criterion, optimizer, device, threshold, task="train", verbose=False):
    """
    evaluate the model
    """

    if task == "test":
        model.eval()
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
            model.eval()
        elif task == "train":
            model.train()
        else:
            raise NameError("Only train, val or test is allowed as task")

    with trange(len(generator), disable=(not verbose)) as t:
        for input_, target, batch_comp, batch_ids in generator:

            # move tensors to GPU
            input_ = (tensor.to(device) for tensor in input_)
            target = target.to(device)
            #print(target)

            # compute output
            output, react_embed = model(*input_)

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
                    #plot_grad_flow(model.named_parameters())
                    optimizer.step()

            t.update()

    if task == "test":
        return test_ids, test_comp, test_pred, test_react_embed, test_targets, subset_accuracy, test_total
    else:
        return loss_meter.avg

