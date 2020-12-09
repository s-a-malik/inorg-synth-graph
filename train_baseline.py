import os
import gc
import sys
import datetime
import pickle as pkl
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split as split
from sklearn.metrics import r2_score, hamming_loss, accuracy_score, f1_score

from matgps.concat_baseline.model import ConcatNet
from matgps.concat_baseline.data import ReactionData

from matgps.utils import save_checkpoint, load_previous_state


def main():

    train_set = ReactionData(
        data_path=args.train_path,
        elem_dict_path=args.elem_path,
        prec_type=args.prec_type,
        augment=args.augment
    )

    embedd_dim = train_set.embedd_dim
    max_prec = train_set.max_prec

    train_idx = list(range(len(train_set)))
    train_set = torch.utils.data.Subset(train_set, train_idx[0::args.sample])

    test_set = ReactionData(
        data_path=args.test_path,
        elem_dict_path=args.elem_path,
        prec_type=args.prec_type,
        augment=args.augment
    )

    if args.val_path:
        val_set = ReactionData(
            data_path=args.val_path,
            elem_dict_path=args.elem_path,
            prec_type=args.prec_type,
            augment=args.augment
        )
    else:
        print("No validation set gvien, using test set for evaluation purposes")
        val_set = test_set

    # Ensure directory structure present
    os.makedirs("models/", exist_ok=True)
    os.makedirs("runs/", exist_ok=True)
    os.makedirs("results/", exist_ok=True)

    print("Shape of train set, test set: ", np.shape(train_set), np.shape(test_set))

    if args.get_reaction_emb:
        get_reaction_emb(args.fold_id, args.ensemble, train_set, max_prec, embedd_dim, "train")
        get_reaction_emb(args.fold_id, args.ensemble, test_set, max_prec, embedd_dim, "test")
        if args.val_path:
            get_reaction_emb(args.fold_id, args.ensemble, val_set, max_prec, embedd_dim, "val")
        return

    if not args.evaluate:
        train_ensemble(args.fold_id, train_set, val_set, args.ensemble, max_prec, embedd_dim)

    test_ensemble(args.fold_id, args.ensemble, test_set, max_prec, embedd_dim)


def train_ensemble(
    fold_id,
    train_set,
    val_set,
    ensemble_folds,
    max_prec,
    embedd_dim
):
    """
    Train multiple models
    """

    params = {"batch_size": args.batch_size,
              "num_workers": args.workers,
              "pin_memory": False,
              "shuffle": True}

    train_generator = DataLoader(train_set, **params)
    val_generator = DataLoader(val_set, **params)

    for run_id in range(ensemble_folds):
        # this allows us to run ensembles in parallel rather than in series
        # by specifiying the run-id arg.
        if ensemble_folds == 1:
            run_id = args.run_id

        model = init_model(
            max_prec=max_prec,
            embedd_dim=embedd_dim,
            intermediate_dim=args.intermediate_dim,
            target_dim=args.target_dim,
            mask=args.mask,
            device=args.device
        )
        criterion, optimizer, scheduler = init_optim(model)

        if args.log:
            writer = SummaryWriter(log_dir=("runs/f-{f}_r-{r}_s-{s}_t-{t}_"
                                            "{date:%d-%m-%Y_%H:%M:%S}").format(
                                                date=datetime.datetime.now(),
                                                f=fold_id,
                                                r=run_id,
                                                s=args.seed,
                                                t=args.sample))
        else:
            writer = None

        experiment(fold_id, run_id, args,
                    train_generator, val_generator,
                    model, optimizer, criterion, scheduler, writer)


def experiment(fold_id, run_id, args,
               train_generator, val_generator,
               model, optimizer, criterion, scheduler, writer):
    """
    for given training and validation sets run an experiment.
    """

    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total Number of Trainable Parameters: {}".format(num_param))

    checkpoint_file = (f"models/checkpoint_f-{fold_id}_r-{run_id}_s-{args.seed}_t-{args.sample}.pth.tar")
    best_file = (f"models/best_f-{fold_id}_r-{run_id}_s-{args.seed}_t-{args.sample}.pth.tar")

    if args.resume:
        print("Resume Training from previous model")
        previous_state = load_previous_state(
            checkpoint_file,
            model,
            args.device,
            optimizer,
            scheduler
        )
        model, optimizer, scheduler, best_loss, start_epoch = previous_state
        model.to(args.device)
    else:
        if args.fine_tune:
            print("Fine tune from a network trained on a different dataset")
            previous_state = load_previous_state(args.fine_tune,
                                                 model,
                                                 args.device)
            model, _, _, _, _ = previous_state
            model.to(args.device)
            criterion, optimizer, scheduler = init_optim(model)
        elif args.transfer:
            print("Use model as an element selector and predict stoichiometry")
            previous_state = load_previous_state(args.transfer,
                                                 model,
                                                 args.device)
            model, _, _, _, _ = previous_state
            # stops autograd from changing parameters of trained network
            for p in model.parameters():
                p.requires_grad = False
            criterion, optimizer, scheduler = init_optim(model)

        best_loss = model.evaluate(
            generator=val_generator,
            criterion=criterion,
            optimizer=None,
            device=args.device,
            threshold=args.threshold,
            task="val"
        )
        start_epoch = 0

    # try except structure used to allow keyboard interupts to stop training
    # without breaking the code
    try:
        for epoch in range(start_epoch, start_epoch+args.epochs):
            # Training
            t_loss = model.evaluate(
                generator=train_generator,
                criterion=criterion,
                optimizer=optimizer,
                device=args.device,
                threshold=args.threshold,
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
                    threshold=args.threshold,
                    task="val"
                )

            # if epoch % args.print_freq == 0:
            print("Epoch: [{}/{}]\n"
                  "Train      : Loss {:.4f}\t"
                  "Validation : Loss {:.4f}\t".format(
                    epoch+1, start_epoch + args.epochs,
                    t_loss, val_loss))

            is_best = val_loss < best_loss
            if is_best:
                best_loss = val_loss

            checkpoint_dict = {"epoch": epoch,
                               "state_dict": model.state_dict(),
                               "best_error": best_loss,
                               "optimizer": optimizer.state_dict(),
                               "scheduler": scheduler.state_dict(),
                               "args": vars(args)}

            save_checkpoint(checkpoint_dict,
                            is_best,
                            checkpoint_file,
                            best_file)

            if args.log:
                writer.add_scalar("loss/train", t_loss, epoch+1)
                writer.add_scalar("loss/validation", val_loss, epoch+1)

            scheduler.step()

            # catch memory leak
            gc.collect()

    except KeyboardInterrupt:
        pass

    if args.log:
        writer.close()


def test_ensemble(fold_id, ensemble_folds, hold_out_set, max_prec, embedd_dim):
    """
    take an ensemble of models and evaluate their performance on the test set
    """

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
          "------------Evaluate model on Test Set------------\n"
          "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    model = init_model(
        max_prec=max_prec,
        embedd_dim=embedd_dim,
        intermediate_dim=args.intermediate_dim,
        target_dim=args.target_dim,
        mask=args.mask,
        device=args.device
    )

    criterion, _, _, = init_optim(model)

    params = {"batch_size": args.batch_size,
              "num_workers": args.workers,
              "pin_memory": False,
              "shuffle": False}

    test_generator = DataLoader(hold_out_set, **params)

    y_ensemble = np.zeros((ensemble_folds, len(hold_out_set), args.target_dim))
    y_ensemble_prec_embed = np.zeros((ensemble_folds, len(hold_out_set), max_prec*embedd_dim))

    for j in range(ensemble_folds):

        if ensemble_folds == 1:
            j = args.run_id
            print("Evaluating Model")
        else:
            print(f"Evaluating Model {j+1}/{ensemble_folds}")

        # checkpoint = torch.load(f=("models/best_"
        checkpoint = torch.load(f=("models/checkpoint_"
                                   f"f-{fold_id}_r-{j}_s-{args.seed}_t-{args.sample}.pth.tar"),
                                map_location=args.device)
        model.load_state_dict(checkpoint["state_dict"])

        model.eval()
        idx, pred, prec_embed, y_test, subset_accuracy, total = model.evaluate(
            generator=test_generator,
            criterion=criterion,
            optimizer=None,
            device=args.device,
            threshold=args.threshold,
            task="test"
        )

        y_ensemble[j, :] = pred
        y_ensemble_prec_embed[j, :] = prec_embed

    y_pred = np.mean(y_ensemble, axis=0)
    y_prec_embed = np.mean(y_ensemble_prec_embed, axis=0)
    y_test = np.array(y_test)
    # print(y_pred[:5])
    # print(y_prec_embed[:5])
    # print(y_test[:5])
    print("Ensemble Performance Metrics:")
    print(f"Elements Accuracy on {total} images (in final ensemble!): {subset_accuracy/total}")

    # thresholds for element prediction
    thresholds = np.linspace(0.005, 0.99, 100)
    subset_acc_dict = {}
    hamming_dict = {}
    f1 = {}
    test_elems = y_test != 0                # bool 2d array
    for threshold in thresholds:
        logit_threshold = np.log(threshold / (1 - threshold))
        pred_elems = y_pred > logit_threshold
        # metrics:
        subset_acc_dict[threshold] = accuracy_score(test_elems, pred_elems)
        f1[threshold] = f1_score(test_elems, pred_elems, average='weighted', zero_division=0)
        hamming_dict[threshold] = hamming_loss(test_elems, pred_elems)

    max_acc = max(subset_acc_dict.values())
    best_subset_acc = [{k: v} for k, v in subset_acc_dict.items() if v == max_acc]
    best_thresh = list(best_subset_acc[0].keys())[0]
    # print(best_thresh)
    best_logit_thresh = np.log(best_thresh / (1 - best_thresh))

    # get uncertainty estimate from ensemble for best_subset_acc threshold
    ensemble_accs = [accuracy_score(test_elems, pred_logits > best_logit_thresh) for pred_logits in y_ensemble]
    ensemble_error = np.std(ensemble_accs)

    print(f"Best Subset 0/1 score: {max_acc} +/- {ensemble_error} at {best_thresh}")
    # print("ensemble accuracies", ensemble_accs)
    # print("Subset 0/1 score", subset_acc_dict)
    # print("Hamming Loss:", hamming_dict)
    # print("F1 Score (weighted):", f1)

    print("y_pred", np.shape(y_pred))
    print("y_prec_embed", np.shape(y_prec_embed))
    print("y_test", np.shape(y_test))
    print("idx", np.shape(idx))

    df = pd.DataFrame()
    df["idx"] = idx  # convert tensor to number
    df["y_test"] = [list(a) for a in y_test]  # make into list of vectors
    df["y_pred"] = [list(a) for a in y_pred]
    df.to_csv(
        index=False,
        path_or_buf=(
            f"results/test_results_ele_f-{fold_id}_r-{args.run_id}_s-{args.seed}_t-{args.sample}.csv"
        )
    )


def get_reaction_emb(fold_id, ensemble_folds, hold_out_set, max_prec, embedd_dim, set_name):
    model = init_model(
        max_prec=max_prec,
        embedd_dim=embedd_dim,
        intermediate_dim=args.intermediate_dim,
        target_dim=args.target_dim,
        mask=args.mask,
        device=args.device
    )

    criterion, _, _, = init_optim(model)

    params = {"batch_size": args.batch_size,
              "num_workers": args.workers,
              "pin_memory": False,
              "shuffle": False}

    test_generator = DataLoader(hold_out_set, **params)

    y_ensemble = np.zeros((ensemble_folds, len(hold_out_set), args.target_dim))
    y_ensemble_prec_embed = np.zeros((ensemble_folds, len(hold_out_set), max_prec*embedd_dim))

    for j in range(ensemble_folds):

        if ensemble_folds == 1:
            j = args.run_id
            print("Evaluating Model")
        else:
            print(f"Evaluating Model {j+1}/{ensemble_folds}")

        # checkpoint = torch.load(f=("models/best_"
        checkpoint = torch.load(f=("models/checkpoint_"
                                   f"f-{fold_id}_r-{j}_s-{args.seed}_t-{args.sample}.pth.tar"),
                                map_location=args.device)
        model.load_state_dict(checkpoint["state_dict"])

        model.eval()
        idx, pred, prec_embed, y_test, subset_accuracy, total = model.evaluate(
            generator=test_generator,
            criterion=criterion,
            optimizer=None,
            device=args.device,
            threshold=args.threshold,
            task="test"
        )

        y_ensemble[j, :] = pred
        y_ensemble_prec_embed[j, :] = prec_embed

    y_pred = np.mean(y_ensemble, axis=0)
    y_prec_embed = np.mean(y_ensemble_prec_embed, axis=0)
    y_test = np.array(y_test)

    results = [y_pred, y_test, y_prec_embed, idx]
    with open(f"data/{set_name}_f-{fold_id}_emb_baseline.pkl", 'wb') as f:
        pkl.dump(results, f)
    print(f'Dumped logits, targets, prec_embeddings, and ids to results file')


def custom_loss(output, target_labels):
    """
    loss function with number of elements predictor, weighted by elements in batch.
    Includes a regularisation term as well
    Hyperparameters: reg_weight - weighting given to regularisation loss function
    """

    # get weights of elements in batch
    num_elements = float(len(target_labels)) - (target_labels == 0).sum(dim=0)
    max_num_elements = torch.max(num_elements)
    max_num_elements_tensor = max_num_elements.repeat(len(num_elements))
    pos_weight = torch.where(num_elements != 0, max_num_elements_tensor / num_elements, num_elements)
    # print(pos_weight)

    # BCE loss for composition - treating as a multilabel classification problem
    comp_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(output, target_labels)

    # L1 regularisation of number of elements in composition for sparsity
    reg_loss = torch.norm(output, 1)

    return comp_loss + (args.reg_weight*reg_loss)


def init_model(max_prec, embedd_dim, intermediate_dim, target_dim, mask, device):
    """Initialise model"""

    model = ConcatNet(
        max_prec=max_prec,
        embedding_dim=embedd_dim,
        intermediate_dim=intermediate_dim,
        target_dim=target_dim,
        mask=mask
    )

    model.to(device)
    print(model)

    return model


def init_optim(model):

    # Select Loss Function, Note we use Robust loss functions that
    # are used to train an aleatoric error estimate
    if args.loss == "BCE":
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == "Custom":
        criterion = custom_loss
    else:
        raise NameError("Only custom or BCE are allowed as --loss")

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

    if args.clr:
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=args.learning_rate/10,
            max_lr=args.learning_rate,
            step_size_up=50,
            cycle_momentum=False)
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [])

    return criterion, optimizer, scheduler


def input_parser():
    """
    parse input
    """
    parser = argparse.ArgumentParser(description="Inorganic Reaction Product Predictor, baseline model")

    # dataset inputs
    parser.add_argument("--train-path",
                        type=str,
                        default="data/train_10_precs.pkl",
                        metavar="PATH",
                        help="Path to results dataframe from element prediction")

    parser.add_argument("--test-path",
                        type=str,
                        default="data/test_10_precs.pkl",
                        metavar="PATH",
                        help="Path to results dataframe from element prediction")

    parser.add_argument("--val-path",
                        type=str,
                        default=None,
                        metavar="PATH",
                        help="Path to results dataframe from element prediction")

    parser.add_argument('--elem-path',
                        type=str,
                        nargs='?',
                        default='data/elem_dict_10_precs.json',
                        help="Path to element dictionary")

    parser.add_argument('--prec-type',
                        type=str,
                        nargs='?',
                        default='magpie',
                        help="Type of input, stoich or magpie")

    parser.add_argument('--intermediate-dim',
                        type=int,
                        nargs='?',
                        default=256,
                        help='Intermediate model dimension')

    parser.add_argument('--target-dim',
                        type=int,
                        nargs='?',
                        default=81,
                        help='Target vector dimension')

    parser.add_argument('--mask',
                        action="store_true",
                        default=False,
                        help="Whether to mask output with precursor elements or not")

    parser.add_argument("--disable-cuda",
                        action="store_true",
                        help="Disable CUDA")

    # restart inputs
    parser.add_argument("--evaluate",
                        action="store_true",
                        help="skip network training and stages checkpoint")

    # dataloader inputs
    parser.add_argument("--workers",
                        default=0,
                        type=int,
                        metavar="N",
                        help="number of data loading workers (default: 0)")

    parser.add_argument("--batch-size", "--bsize",
                        default=64,
                        type=int,
                        metavar="N",
                        help="mini-batch size")

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

    # optimiser inputs
    parser.add_argument("--epochs",
                        default=100,
                        type=int,
                        metavar="N",
                        help="number of total epochs to run")

    parser.add_argument("--loss",
                        default="BCE",
                        type=str,
                        metavar="str",
                        help="choose a Loss Function")

    parser.add_argument("--threshold",
                        default=0.5,
                        type=float,
                        metavar='float [0,1]',
                        help="Threshold for element presence in product (probability)")

    parser.add_argument("--reg-weight",
                        default=0,
                        type=float,
                        metavar="float",
                        help="Weight for regularisation loss")

    parser.add_argument("--optim",
                        default="Adam",
                        type=str,
                        metavar="str",
                        help="choose an optimizer; SGD, Adam or AdamW")

    parser.add_argument("--learning-rate", "--lr",
                        default=0.0001,
                        type=float,
                        metavar="float",
                        help="initial learning rate (default: 3e-4)")

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

    # ensemble inputs
    parser.add_argument("--fold-id",
                        default=1,
                        type=int,
                        metavar="N",
                        help="identify the fold of the data")

    parser.add_argument("--run-id",
                        default=0,
                        type=int,
                        metavar="N",
                        help="ensemble model id")

    parser.add_argument("--ensemble",
                        default=1,
                        type=int,
                        metavar="N",
                        help="number ensemble repeats")

    parser.add_argument("--augment",
                        action="store_true",
                        help="augment ordering of precursors")

    # transfer learning
    parser.add_argument("--clr",
                        default=True,
                        type=bool,
                        help="use a cyclical learning rate schedule")

    parser.add_argument("--resume",
                        action="store_true",
                        help="resume from previous checkpoint")

    parser.add_argument("--log",
                        action="store_true",
                        help="write tensorboard logs")

    parser.add_argument("--transfer",
                        type=str,
                        metavar="PATH",
                        help="checkpoint path for transfer learning")

    parser.add_argument("--fine-tune",
                        type=str,
                        metavar="PATH",
                        help="checkpoint path for fine tuning")

    parser.add_argument("--get-reaction-emb",
                        action="store_true",
                        help="resume from previous checkpoint")

    args = parser.parse_args(sys.argv[1:])

    args.device = torch.device("cuda") if (not args.disable_cuda) and  \
        torch.cuda.is_available() else torch.device("cpu")

    return args


if __name__ == "__main__":
    args = input_parser()

    print("The model will run on the {} device".format(args.device))

    main()
