import os
import gc
import sys
import datetime
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split as split
from sklearn.metrics import r2_score

from matgps.stoich.model import StoichNet
from matgps.stoich.data import ProductData, collate_batch

from matgps.utils import save_checkpoint, load_previous_state


def init_model(orig_atom_fea_len, orig_reaction_fea_len):

    model = StoichNet(
        orig_elem_fea_len=orig_atom_fea_len,
        orig_reaction_fea_len=orig_reaction_fea_len,
        intermediate_dim=args.intermediate_dim,
        n_heads=args.n_heads
    )

    print(model)
    model.to(args.device)

    return model


def init_optim(model):

    # Select Loss Function
    if args.loss == "MSE":
        criterion = nn.MSELoss()
    elif args.loss == "MAE":
        criterion = nn.L1Loss()
    else:
        raise NameError("Only MSE or MAE are allowed as --loss")

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


def main():

    dataset = ProductData(
        data_path=args.data_path,
        fea_path=args.elem_fea_path,
        elem_path=args.elem_path,
        threshold=args.threshold,
        use_correct_targets=args.use_correct_targets
    )

    orig_atom_fea_len = dataset.atom_fea_dim    # atom embedding dimension
    orig_reaction_fea_len = dataset.reaction_fea_dim    # reaction embedding dimension
    print('orig atom embedding dimension', orig_atom_fea_len)
    print('orig reaction embedding dimension', orig_reaction_fea_len)

    # skip to evaluate whole dataset if testing all
    if args.test_size == 1.0:

        test_ensemble(args.fold_id, args.ensemble, dataset, orig_atom_fea_len, orig_reaction_fea_len)
        return

    indices = list(range(len(dataset)))
    train_idx, test_idx = split(indices, random_state=args.seed,
                                test_size=args.test_size)

    train_set = torch.utils.data.Subset(dataset, train_idx[0::args.sample])
    test_set = torch.utils.data.Subset(dataset, test_idx)

    # Ensure directory structure present
    os.makedirs(f"models/", exist_ok=True)
    os.makedirs("runs/", exist_ok=True)
    os.makedirs("results/", exist_ok=True)

    print("Shape of train set, test set: ", np.shape(train_set), np.shape(test_set))

    ensemble(args.fold_id, train_set, test_set,
             args.ensemble, orig_atom_fea_len, orig_reaction_fea_len)


def ensemble(fold_id, dataset, test_set,
             ensemble_folds, fea_len, reaction_fea_len):
    """
    Train multiple models
    """

    params = {"batch_size": args.batch_size,
              "num_workers": args.workers,
              "pin_memory": False,
              "shuffle": True,
              "collate_fn": collate_batch}

    if args.val_size == 0.0:
        print("No validation set used, using test set for evaluation purposes")
        # NOTE that when using this option care must be taken not to
        # peak at the test-set. The only valid model to use is the one obtained
        # after the final epoch where the epoch count is decided in advance of
        # the experiment.
        train_subset = dataset
        val_subset = test_set
    else:
        indices = list(range(len(dataset)))
        train_idx, val_idx = split(indices, random_state=args.seed,
                                   test_size=args.val_size/(1-args.test_size))
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)
        print("Shape of train, val subset: ", np.shape(train_subset), np.shape(val_subset))

    train_generator = DataLoader(train_subset, **params)
    val_generator = DataLoader(val_subset, **params)

    if not args.evaluate:
        for run_id in range(ensemble_folds):

            # this allows us to run ensembles in parallel rather than in series
            # by specifiying the run-id arg.
            if ensemble_folds == 1:
                run_id = args.run_id

            model = init_model(fea_len, reaction_fea_len)
            criterion, optimizer, scheduler = init_optim(model)
            writer = SummaryWriter(log_dir=("runs/f-{f}_r-{r}_s-{s}_t-{t}_"
                                            "{date:%d-%m-%Y_%H:%M:%S}").format(
                                                date=datetime.datetime.now(),
                                                f=fold_id,
                                                r=run_id,
                                                s=args.seed,
                                                t=args.sample))

            experiment(fold_id, run_id, args,
                       train_generator, val_generator,
                       model, optimizer, criterion, scheduler, writer)

    test_ensemble(fold_id, ensemble_folds, test_set, fea_len, reaction_fea_len)


def experiment(fold_id, run_id, args,
               train_generator, val_generator,
               model, optimizer, criterion, scheduler, writer):
    """
    for given training and validation sets run an experiment.
    """

    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total Number of Trainable Parameters: {}".format(num_param))

    checkpoint_file = ("models/checkpoint_"
                       "f-{}_r-{}_s-{}_t-{}.pth.tar").format(fold_id,
                                                             run_id,
                                                             args.seed,
                                                             args.sample)
    best_file = ("models/best_"
                 "f-{}_r-{}_s-{}_t-{}.pth.tar").format(fold_id,
                                                       run_id,
                                                       args.seed,
                                                       args.sample)

    if args.resume:
        print("Resume Training from previous model")
        previous_state = load_previous_state(checkpoint_file,
                                             model,
                                             args.device,
                                             optimizer,
                                             scheduler)
        model, optimizer, scheduler, \
            best_loss, start_epoch = previous_state
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
            print("Use model as a feature extractor and retrain last layer")
            previous_state = load_previous_state(args.transfer,
                                                 model,
                                                 args.device)
            model, _, _, _, _ = previous_state
            for p in model.parameters():
                p.requires_grad = False
            num_ftrs = model.output_nn.fc_out.in_features
            model.output_nn.fc_out = nn.Linear(num_ftrs, 2)
            model.to(args.device)
            criterion, optimizer, scheduler = init_optim(model)
            num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print("Total Number of Trainable Parameters: {}".format(num_param))


        best_loss, _, _ = model.evaluate(
            generator=val_generator,
            criterion=criterion,
            optimizer=None,
            device=args.device,
            task="val"
        )
        start_epoch = 0

    # try except structure used to allow keyboard interupts to stop training
    # without breaking the code
    try:
        for epoch in range(start_epoch, start_epoch+args.epochs):
            # Training
            t_loss, t_mae, t_rmse = model.evaluate(
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
                val_loss, val_mae, val_rmse = model.evaluate(
                    generator=val_generator,
                    criterion=criterion,
                    optimizer=None,
                    device=args.device,
                    task="val"
                )

            # if epoch % args.print_freq == 0:
            print("Epoch: [{}/{}]\n"
                  "Train      : Loss {:.4f}\t"
                  "MAE {:.3f}\t RMSE {:.3f}\n"
                  "Validation : Loss {:.4f}\t"
                  "MAE {:.3f}\t RMSE {:.3f}\n".format(
                    epoch+1, start_epoch + args.epochs,
                    t_loss, t_mae, t_rmse,
                    val_loss, val_mae, val_rmse))

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

            writer.add_scalar("loss/train", t_loss, epoch+1)
            writer.add_scalar("loss/validation", val_loss, epoch+1)
            writer.add_scalar("rmse/train", t_rmse, epoch+1)
            writer.add_scalar("rmse/validation", val_rmse, epoch+1)
            writer.add_scalar("mae/train", t_mae, epoch+1)
            writer.add_scalar("mae/validation", val_mae, epoch+1)

            scheduler.step()

            # catch memory leak
            gc.collect()

    except KeyboardInterrupt:
        pass

    writer.close()


def test_ensemble(fold_id, ensemble_folds, hold_out_set, fea_len, reaction_fea_len):
    """
    take an ensemble of models and evaluate their performance on the test set
    """

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
          "------------Evaluate model on Test Set------------\n"
          "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    model = init_model(fea_len, reaction_fea_len)

    criterion, _, _, = init_optim(model)

    params = {"batch_size": args.batch_size,
              "num_workers": args.workers,
              "pin_memory": False,
              "shuffle": False,
              "collate_fn": collate_batch}

    test_generator = DataLoader(hold_out_set, **params)

    y_ensemble = []

    for j in range(ensemble_folds):

        if ensemble_folds == 1:
            j = args.run_id
            print("Evaluating Model")
        else:
            print("Evaluating Model {}/{}".format(j+1, ensemble_folds))

        checkpoint = torch.load(f=("models/checkpoint_"
                                   "f-{}_r-{}_s-{}_t-{}"
                                   ".pth.tar").format(fold_id,
                                                      j,
                                                      args.seed,
                                                      args.sample),
                                map_location=args.device)
        model.load_state_dict(checkpoint["state_dict"])

        model.eval()

        reaction_idx, comp, pred, y_test, total = model.evaluate(
            generator=test_generator,
            criterion=criterion,
            optimizer=None,
            device=args.device,
            task="test"
        )

        y_ensemble.append(pred)

    y_pred = np.mean(y_ensemble, axis=0)
    y_test = np.array(y_test)

    # calculate metrics
    ae = np.abs(y_test - y_pred)
    mae_avg = np.mean(ae)
    mae_std = np.std(ae)/np.sqrt(len(ae))

    se = np.square(y_test - y_pred)
    mse_avg = np.mean(se)
    mse_std = np.std(se)/np.sqrt(len(se))

    rmse_avg = np.sqrt(mse_avg)
    rmse_std = 0.5 * rmse_avg * mse_std / mse_avg


    print("y_pred", np.shape(y_pred))
    print("y_test", np.shape(y_test))
    print(y_pred[:5])
    print(y_test[:5])
    print("Ensemble Performance Metrics:")
    print("R2 Score: {:.4f} ".format(r2_score(y_test, y_pred)))
    print("MAE (over whole vector): {:.4f} +/- {:.4f}".format(mae_avg, mae_std))
    print("RMSE (over whole vector): {:.4f} +/- {:.4f}".format(rmse_avg, rmse_std))

    # seperate into reactions
    y_test_reaction = defaultdict(list)
    y_pred_ensemble = defaultdict(lambda: defaultdict(list))
    y_pred_reaction = defaultdict(list)
    base_id = 0
    for i, elems in enumerate(comp):
        for elem in range(len(elems)):
            y_test_reaction[i].append(y_test[elem+base_id])
            y_pred_reaction[i].append(y_pred[elem+base_id])
            for num in range(len(y_ensemble)):
                y_pred_ensemble[num][i].append(y_ensemble[num][elem+base_id])
        base_id += len(elems)

    # save results
    core = {"id": reaction_idx, "composition": comp}
    results = {"pred-{}".format(num): pd.Series(preds) for (num, preds)
                in y_pred_ensemble.items()}
    df = pd.DataFrame({**core, **results})
    df["pred-ens"] = pd.Series(y_pred_reaction)
    df["target"] = pd.Series(y_test_reaction)
    print(df)

    if ensemble_folds == 1:
        df.to_csv(
            index=False,
            path_or_buf=(
                f"results/test_results_comp_f-{fold_id}_r-{args.run_id}_s-{args.seed}_t-{args.sample}.csv"
            )
        )
        print(
            f"Dumped results df to results/test_results_comp_f-{fold_id}_r-{args.run_id}_s-{args.seed}_t-{args.sample}.csv"
        )
    else:
        df.to_csv(
            index=False,
            path_or_buf=f"results/ensemble_results_comp_f-{fold_id}_s-{args.seed}_t-{args.sample}.csv"
        )
        print(
            f"Dumped results df to results/ensemble_results_comp_f-{fold_id}_s-{args.seed}_t-{args.sample}.csv"
        )


def input_parser():
    """
    parse input
    """
    parser = argparse.ArgumentParser(
        description="Inorganic Reaction Product Predictor, Stoichiometry prediction"
    )
    # dataset inputs
    parser.add_argument("--data-path",
                        type=str,
                        default="results/test_results_f-0_r-0_s-0_t-1.pkl",
                        metavar="PATH",
                        help="Path to results dataframe from element prediction")

    parser.add_argument("--elem-fea-path",
                        type=str,
                        default="data/embeddings/matscholar-embedding.json",
                        metavar="PATH",
                        help="Path to element features")

    parser.add_argument('--elem-path',
                        type=str,
                        nargs='?',
                        default='data/datasets/elem_dict_10_precs.json',
                        help="Path to element dictionary")

    parser.add_argument('--intermediate-dim',
                        type=int,
                        nargs='?',
                        default=256,
                        help='Intermediate model dimension')

    parser.add_argument("--n-heads",
                        default=5,
                        type=int,
                        metavar="N",
                        help="number of attention heads")

    parser.add_argument("--disable-cuda",
                        action="store_true",
                        help="Disable CUDA")

    # restart inputs
    parser.add_argument("--evaluate",
                        action="store_true",
                        help="skip network training stages checkpoint")

    # dataloader inputs
    parser.add_argument("--workers",
                        default=0,
                        type=int,
                        metavar="N",
                        help="number of data loading workers (default: 0)")

    parser.add_argument("--batch-size", "--bsize",
                        default=256,
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

    parser.add_argument("--use-correct-targets",
                        action="store_true",
                        help="Use correct elements for training")

    # optimiser inputs
    parser.add_argument("--epochs",
                        default=200,
                        type=int,
                        metavar="N",
                        help="number of total epochs to run")

    parser.add_argument("--loss",
                        default="MSE",
                        type=str,
                        metavar="str",
                        help="choose a Loss Function")

    parser.add_argument("--threshold",
                        default=0.5,
                        type=float,
                        metavar='prob',
                        help="Threshold for element presence in product (probability)")

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
                        default=0,
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

    # transfer learning
    parser.add_argument("--lr-search",
                        action="store_true",
                        help="perform a learning rate search")

    parser.add_argument("--clr",
                        default=True,
                        type=bool,
                        help="use a cyclical learning rate schedule")

    parser.add_argument("--clr-period",
                        default=100,
                        type=int,
                        help="how many learning rate cycles to perform")

    parser.add_argument("--resume",
                        action="store_true",
                        help="resume from previous checkpoint")

    parser.add_argument("--transfer",
                        type=str,
                        metavar="PATH",
                        help="checkpoint path for transfer learning")

    parser.add_argument("--fine-tune",
                        type=str,
                        metavar="PATH",
                        help="checkpoint path for fine tuning")

    args = parser.parse_args(sys.argv[1:])

    if args.lr_search:
        args.learning_rate = 1e-8

    args.device = torch.device("cuda") if (not args.disable_cuda) and  \
        torch.cuda.is_available() else torch.device("cpu")

    return args


if __name__ == "__main__":
    args = input_parser()

    print("The model will run on the {} device".format(args.device))

    main()
