"""Module containing RNN autoencoder for action sequences
and its training routines: preprocessing sequences, training, saving
"""

import os
import sys
import shutil
import argparse

import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.model_selection import train_test_split as split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from matgps.action_rnn.model import LSTM
from matgps.action_rnn.data import (
    ActionData,
    collate_batch,
)


def custom_rnn_loss(output, target):
    """Cross Entropy loss function with weights from batch
    Weighting: (Max no. of occurences of any class)/(No. of occurances of class)
    target: tensor (total no. of actions in batch) with integer class labels
    """
    # find number of occurances of each type
    num_actions = []
    for i in range(output.shape[1]):
        num_actions.append((target == i).unsqueeze(0))
    num_actions = torch.cat(num_actions, dim=0).sum(dim=1)
    # print(num_actions)
    # print(target)

    # find max number of actions
    max_num_actions = torch.max(num_actions).repeat(len(num_actions))
    # weights
    weight = torch.where(num_actions != 0, max_num_actions / num_actions, num_actions).float()

    weighted_loss = nn.CrossEntropyLoss(ignore_index=0, weight=weight)(output, target)
    # weighted_loss = nn.CrossEntropyLoss(ignore_index=0)(output, target)

    return weighted_loss


def main():
    """
    Train the action sequence encoder.
    """
    # gets raw action sequences (OHE embedded already)
    dataset = ActionData(args.data_path, args.action_path)
    tokens = "\n\t".join(f"{val} - {key}" for (key, val) in dataset.action_dict.items())
    print(f"Model Tokens:\n\t{tokens}")

    # get train/val/test generators - these form the padded sequences
    indices = list(range(len(dataset)))
    train_idx, test_idx = split(indices, random_state=args.seed,
                                test_size=args.test_size)
    train_set = torch.utils.data.Subset(dataset, train_idx[0::args.sample])
    test_set = torch.utils.data.Subset(dataset, test_idx)
    print("Shape of train, test set: ", train_set.__len__(), test_set.__len__())

    if args.val_size == 0.0:
        print("No validation set used, using test set for evaluation purposes")
        # Note that when using this option care must be taken not to
        # peak at the test-set. The only valid model to use is the one obtained
        # after the final epoch where the epoch count is decided in advance of
        # the experiment.
        train_subset = dataset
        val_subset = test_set
    else:
        indices = list(range(len(train_set)))
        train_idx, val_idx = split(indices, random_state=args.seed,
                                   test_size=args.val_size/(1-args.test_size))
        train_subset = torch.utils.data.Subset(train_set, train_idx)
        val_subset = torch.utils.data.Subset(train_set, val_idx)
        print("Shape of train, val subset: ", train_subset.__len__(), val_subset.__len__())

    params = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": False,
        "shuffle": True,
        "collate_fn": collate_batch
    }

    train_generator = DataLoader(train_subset, **params)
    val_generator = DataLoader(val_subset, **params)

    # initialise model and optimization
    model = LSTM(
        input_dim=dataset.action_fea_dim,
        latent_dim=args.latent_dim,
        device=args.device,
        num_layers=args.num_layers,
        embedding_dim=args.embedding_dim
    )
    model.to(args.device)

    if args.loss == "Custom":
        criterion = custom_rnn_loss
    elif args.loss == "CrossEntropy":
        criterion = nn.CrossEntropyLoss(ignore_index=0)
    elif args.loss == "MSE":
        criterion = nn.MSELoss()
    else:
        raise NameError("Only custom or MSE are allowed as --loss")

    # Select Optimiser
    if args.optim == "SGD":
        optimizer = optim.SGD(model.parameters(),
                              lr=args.learning_rate,
                              weight_decay=args.weight_decay,
                              momentum=args.momentum)
    elif args.optim == "Adam":
        optimizer = optim.Adam(model.parameters(),
                               lr=args.learning_rate,
                               weight_decay=args.weight_decay)
    elif args.optim == "AdamW":
        optimizer = optim.AdamW(model.parameters(),
                                lr=args.learning_rate,
                                weight_decay=args.weight_decay)
    else:
        raise NameError("Only SGD or Adam is allowed as --optim")

    # print model details
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total Number of Trainable Parameters: {}".format(num_param))
    print(model)

    # Ensure directory structure present
    os.makedirs(f"models/", exist_ok=True)
    os.makedirs("runs/", exist_ok=True)
    os.makedirs("results/", exist_ok=True)

    # try except structure used to allow keyboard interupts to stop training
    # without breaking the code
    start_epoch = 0
    if not args.evaluate:
        idx_details = f"f-{args.fold_id}_s-{args.seed}_t-{args.sample}"
        writer = SummaryWriter(
            log_dir=(f"runs/rnn-{idx_details}_{datetime.now():%d-%m-%Y_%H-%M-%S}")
        )

        checkpoint_file = f"models/checkpoint_rnn_{idx_details}.pth.tar"
        best_file = f"models/best_rnn_{idx_details}.pth.tar"

        best_loss = model.evaluate(
            generator=val_generator,
            criterion=criterion,
            optimizer=None,
            device=args.device,
            task="val"
        )

        try:
            for epoch in range(start_epoch, start_epoch+args.epochs):
                # Training
                t_loss = model.evaluate(
                    generator=train_generator,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=args.device,
                    task="train",
                    verbose=True
                )

                # Validation
                with torch.no_grad():
                    # evaluate on validation set
                    val_loss = model.evaluate(
                        generator=val_generator,
                        criterion=criterion,
                        optimizer=None,
                        device=args.device,
                        task="val"
                    )

                # if epoch % args.print_freq == 0:
                print("Epoch: [{}/{}]\n"
                    "Train      : Loss {:.4f}\t"
                    "Validation : Loss {:.4f}\t".format(
                        epoch+1, start_epoch + args.epochs,
                        t_loss, val_loss))

                # save model
                is_best = val_loss < best_loss
                if is_best:
                    best_loss = val_loss

                checkpoint_dict = {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "best_error": best_loss,
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args)
                }
                torch.save(checkpoint_dict, checkpoint_file)

                if is_best:
                    shutil.copyfile(checkpoint_file, best_file)

                writer.add_scalar("loss/train", t_loss, epoch+1)
                writer.add_scalar("loss/validation", val_loss, epoch+1)

        except KeyboardInterrupt:
            pass

    # test set
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
          "------------Evaluate model on Test Set------------\n"
          "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    params = {"batch_size": args.batch_size,
              "num_workers": args.workers,
              "pin_memory": False,
              "shuffle": False,
              "collate_fn": collate_batch}

    test_generator = DataLoader(dataset, **params)

    test_pred, test_lens, test_encoded, test_targets, test_total = model.evaluate(
        generator=test_generator,
        criterion=criterion,
        optimizer=optimizer,
        device=args.device,
        task="test",
        verbose=True
    )

    # test actual sequences
    y_pred_seq = []
    y_target_seq = []
    for reaction in range(len(test_targets)):
        y_pred_seq.append([np.argmax(test_pred[reaction][x]) for x in range(int(test_lens[reaction])-1)])
        y_target_seq.append([np.argmax(test_targets[reaction][x]) for x in range(int(test_lens[reaction])-1)])

    # print(y_pred_seq[:10])
    # print(y_target_seq[:10])

    # save results
    df = pd.DataFrame({"y_pred": y_pred_seq, "y_target": y_target_seq})
    df.to_csv(index=False, path_or_buf=(f"results/rnn_f-{args.fold_id}.csv"))
    print("dumped preds and targets to df")

    # metrics
    correct = 0
    for reaction in range(len(y_pred_seq)):
        if y_target_seq[reaction] == y_pred_seq[reaction]:
            correct += 1
    subset_acc = correct/len(y_target_seq)
    print('subset acc', subset_acc)

    y_pred_all = [item for t in y_pred_seq for item in t]
    y_target_all = [item for t in y_target_seq for item in t]
    diff = np.subtract(y_pred_all, y_target_all)
    accuracy = (len(y_pred_all)-np.count_nonzero(diff))/len(y_pred_all)
    print('total accuracy', accuracy)


def input_parser():
    """
    parse input
    """
    parser = argparse.ArgumentParser(description="Action RNN Autoencoder Training")

    # dataset inputs
    parser.add_argument("--data-path",
                        type=str,
                        default="data/datasets/dataset_10_precs.pkl",
                        metavar="PATH",
                        help="dataset path")

    parser.add_argument("--action-path",
                        type=str,
                        default="data/datasets/action_dict_10_precs.json",
                        metavar="PATH",
                        help="action dict path")

    parser.add_argument("--embedding-dim",
                        type=int,
                        default=8,
                        metavar="N",
                        help="Dim of embedding rep for sequences (linear embedding instead of OHE)")

    parser.add_argument("--latent-dim",
                        type=int,
                        default=32,
                        metavar="N",
                        help="Dim of latent representation of sequence")

    parser.add_argument("--num-layers",
                        type=int,
                        default=1,
                        metavar="N",
                        help="Number of LSTM layers")

    # dataloader inputs
    parser.add_argument("--workers",
                        default=0,
                        type=int,
                        metavar="N",
                        help="number of data loading workers (default: 0)")

    parser.add_argument("--batch-size", "--bsize",
                        default=128,
                        type=int,
                        metavar="N",
                        help="mini-batch size (default: 128)")

    parser.add_argument("--val-size",
                        default=0.0,
                        type=float,
                        metavar="N",
                        help="proportion of data used for validation")

    parser.add_argument("--test-size",
                        default=0.2,
                        type=float,
                        metavar="N",
                        help="proportion of data for testing")

    parser.add_argument("--seed",
                        default=0,
                        type=int,
                        metavar="N",
                        help="seed for random number generator")

    parser.add_argument("--sample",
                        default=1,
                        type=int,
                        metavar="N",
                        help="sub-sample the training set for learning curves")

    parser.add_argument("--fold-id",
                        default=0,
                        type=int,
                        metavar="N",
                        help="fold id for run")

    parser.add_argument("--evaluate",
                        action="store_true",
                        help="skip network training stages checkpoint")

    # optimiser inputs
    parser.add_argument("--epochs",
                        # default=10,
                        default=100,
                        type=int,
                        metavar="N",
                        help="number of total epochs to run")

    parser.add_argument("--loss",
                        default="Custom",
                        type=str,
                        metavar="str",
                        help="choose a Loss Function")

    parser.add_argument("--optim",
                        default="SGD",
                        type=str,
                        metavar="str",
                        help="choose an optimizer; SGD, Adam or AdamW")

    parser.add_argument("--learning-rate", "--lr",
                        default=0.3,
                        type=float,
                        metavar="float",
                        help="initial learning rate (default: 0.3)")

    parser.add_argument("--momentum",
                        default=0.9,
                        type=float,
                        metavar="float [0,1]",
                        help="momentum (default: 0.9)")

    parser.add_argument("--weight-decay",
                        default=1e-6,
                        type=float,
                        metavar="float [0,1]",
                        help="weight decay (default: 0)")

    parser.add_argument('--teacher-forcing',
                        action="store_true",
                        help='If using the teacher frocing in decoder')

    parser.add_argument("--lr-search",
                        action="store_true",
                        help="perform a learning rate search")

    parser.add_argument("--disable-cuda",
                        action="store_true",
                        help="Disable CUDA")

    args = parser.parse_args(sys.argv[1:])

    args.device = torch.device("cuda") if (not args.disable_cuda) and  \
        torch.cuda.is_available() else torch.device("cpu")

    return args


if __name__ == "__main__":

    args = input_parser()
    print(f"The model will run on the {args.device} device")

    main()