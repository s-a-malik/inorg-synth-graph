import os
import torch
from tqdm.autonotebook import trange
import shutil
import math
import numpy as np

import copy

from tqdm.autonotebook import tqdm
from torch.optim.lr_scheduler import _LRScheduler
import matplotlib.pyplot as plt
from matplotlib.pyplot import Line2D

from torch.optim import Optimizer
from torch.nn.functional import l1_loss as mae
from torch.nn.functional import mse_loss as mse
from stoich.data import AverageMeter


def evaluate(generator, model, criterion, optimizer, device, task="train", verbose=False):
    """
    evaluate the model
    """

    if task == "test":
        model.eval()
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

            # compute output
            output = model(*input_)

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
                    #plot_grad_flow(model.named_parameters())
                    optimizer.step()
            t.update()

    if task == "test":
        return test_ids, test_comp, test_pred, test_targets, test_total
    else:
        return loss_meter.avg, mae_meter.avg, rmse_meter.avg
