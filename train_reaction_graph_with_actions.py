import os
import gc
import datetime
import pickle as pkl

import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split as split
from sklearn.metrics import (
    r2_score,
    hamming_loss,
    accuracy_score,
    f1_score,
)

from reaction_graph_actions.model import ReactionNet
from reaction_graph_actions.action_rnn import LSTM
from reaction_graph_actions.data import (
    input_parser,
    ReactionData,
    collate_batch,
)
from reaction_graph_actions.utils import (
    evaluate,
    save_checkpoint,
    load_previous_state,
    cyclical_lr,
    LRFinder
)


def get_class_weights(generator):
    """Get class weights for imbalanced dataset by iterating through generator once
    """
    all_targets = []
    for _, target, _, _ in generator:
        all_targets += target.tolist()
    # print(all_targets[:5])
    all_targets = torch.Tensor(all_targets)
    # print(all_targets.shape)
    # get weights of elements in batch
    num_elements = float(len(all_targets)) - (all_targets == 0).sum(dim=0)
    # print(num_elements)
    max_num_elements = torch.max(num_elements)
    max_num_elements_tensor = max_num_elements.repeat(len(num_elements))
    # print(max_num_elements_tensor)
    weights = torch.where(num_elements != 0, max_num_elements_tensor / num_elements, num_elements)
    print(weights)

    return weights.to(args.device)


def custom_loss(output, target_labels):
    """loss function with batchwise weighting and regularisation
    Hyperparameters: reg_weight - weighting given to regularisation loss function
    """

    # get weights of elements in batch
    num_elements = float(len(target_labels)) - (target_labels == 0).sum(dim=0)
    max_num_elements = torch.max(num_elements)
    max_num_elements_tensor = max_num_elements.repeat(len(num_elements))
    pos_weight = torch.where(num_elements != 0, max_num_elements_tensor / num_elements, num_elements)
    #print(pos_weight)

    # BCE loss for composition - treating as a multilabel classification problem
    comp_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(output, target_labels)

    # L1 regularisation of number of elements in composition for sparsity
    reg_loss = torch.norm(output, 1)

    return comp_loss + (args.reg_weight*reg_loss)


def init_model(pretrained_rnn, orig_prec_fea_len):

    model = ReactionNet(
        pretrained_rnn=pretrained_rnn,
        orig_prec_fea_len=orig_prec_fea_len,
        prec_fea_len=args.prec_fea_len,
        n_graph=args.n_graph,
        intermediate_dim=args.intermediate_dim,
        target_dim=args.target_dim,
        mask=args.mask
    )

    model.to(args.device)
    print(model)

    return model


def init_optim(model, weights=None):

    # Select Loss Function, Note we use Robust loss functions that
    # are used to train an aleatoric error estimate
    if args.loss == "BCE":
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == "BCEweighted":
        if weights == None:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
    elif args.loss == "custom":
        criterion = custom_loss
    else:
        raise NameError("Only custom or MAE are allowed as --loss")

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
        clr = cyclical_lr(period=args.clr_period,
                          cycle_mul=0.1,
                          tune_mul=0.05,)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, [clr])
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [])

    return criterion, optimizer, scheduler


def main():

    dataset = ReactionData(
        data_path=args.data_path,
        fea_path=args.fea_path,
        action_dict_path=args.action_path,
        elem_dict_path=args.elem_path,
        prec_type=args.prec_type,
        amounts=args.amounts
    )

    orig_prec_fea_len = dataset.prec_fea_dim
    orig_action_fea_len = dataset.action_fea_dim
    print('orig precursor fea dim', orig_prec_fea_len)
    print('action fea dim', orig_action_fea_len)

    # load pretrained rnn
    rnn = LSTM(
        input_dim=orig_action_fea_len,
        latent_dim=args.latent_dim,
        device=args.device,
        num_layers=1,
        embedding_dim=8
    )

    if args.action_rnn:
        print('loading pretrained RNN...')
        previous_state = load_previous_state(
            path=args.action_rnn,
            model=rnn,
            device=args.device
        )
        pretrained_rnn, _, _, _, _ = previous_state

    if not args.train_rnn:
        # stops autograd from changing parameters of trained network
        for p in pretrained_rnn.parameters():
            p.requires_grad = False

    # skip to evaluate whole dataset if testing all
    if args.test_size == 1.0:
        test_ensemble(args.fold_id, args.ensemble, dataset, orig_prec_fea_len, pretrained_rnn)
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
             args.ensemble, orig_prec_fea_len, pretrained_rnn)


def ensemble(fold_id, dataset, test_set,
             ensemble_folds, fea_len, pretrained_rnn):
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
        # Note that when using this option care must be taken not to
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
    weights = get_class_weights(train_generator)

    if not args.evaluate:
        if args.lr_search:
            model = init_model(pretrained_rnn, fea_len)
            criterion, optimizer, scheduler = init_optim(model, weights=weights)

            if args.fine_tune:
                print("Fine tune from a network trained on a different dataset")
                previous_state = load_previous_state(args.fine_tune,
                                                     model,
                                                     args.device)
                model, _, _, _, _ = previous_state
                model.to(args.device)
                criterion, optimizer, scheduler = init_optim(model, weights=weights)

            lr_finder = LRFinder(model, optimizer, criterion,
                                 metric="mse", device=args.device)
            lr_finder.range_test(train_generator, end_lr=1,
                                 num_iter=100, step_mode="exp")
            lr_finder.plot()
            return

        for run_id in range(ensemble_folds):

            # this allows us to run ensembles in parallel rather than in series
            # by specifiying the run-id arg.
            if ensemble_folds == 1:
                run_id = args.run_id

            model = init_model(pretrained_rnn, fea_len)
            criterion, optimizer, scheduler = init_optim(model, weights=weights)

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

    test_ensemble(fold_id, ensemble_folds, test_set, fea_len, pretrained_rnn)


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
            print("Use model as an element selector and predict stoichiometry")
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


        best_loss = evaluate(generator=val_generator,
                                  model=model,
                                  criterion=criterion,
                                  optimizer=None,
                                  device=args.device,
                                  threshold=args.threshold,
                                  task="val")
        start_epoch = 0

    # try except structure used to allow keyboard interupts to stop training
    # without breaking the code
    try:
        for epoch in range(start_epoch, start_epoch+args.epochs):
            # Training
            t_loss = evaluate(generator=train_generator,
                                             model=model,
                                             criterion=criterion,
                                             optimizer=optimizer,
                                             device=args.device,
                                             threshold=args.threshold,
                                             task="train",
                                             verbose=True)

            # Validation
            with torch.no_grad():
                # evaluate on validation set
                val_loss = evaluate(generator=val_generator,
                                                       model=model,
                                                       criterion=criterion,
                                                       optimizer=None,
                                                       device=args.device,
                                                       threshold=args.threshold,
                                                       task="val")

            # if epoch % args.print_freq == 0:
            print("Epoch: [{}/{}]\n"
                  "Train      : Loss {:.4f}\t"
                  "Validation : Loss {:.4f}\t".format(
                    epoch+1, start_epoch + args.epochs,
                    t_loss, val_loss,))

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

            scheduler.step()

            # catch memory leak
            gc.collect()

    except KeyboardInterrupt:
        pass

    writer.close()

def test_ensemble(fold_id, ensemble_folds, hold_out_set, fea_len, pretrained_rnn):
    """
    take an ensemble of models and evaluate their performance on the test set
    """

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
          "------------Evaluate model on Test Set------------\n"
          "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    model = init_model(pretrained_rnn, fea_len)

    criterion, _, _, = init_optim(model)

    params = {"batch_size": args.batch_size,
              "num_workers": args.workers,
              "pin_memory": False,
              "shuffle": False,
              "collate_fn": collate_batch}

    test_generator = DataLoader(hold_out_set, **params)

    y_ensemble = np.zeros((ensemble_folds, len(hold_out_set), args.target_dim))
    y_ensemble_prec_embed = np.zeros((ensemble_folds, len(hold_out_set), args.prec_fea_len))

    for j in range(ensemble_folds):

        if ensemble_folds == 1:
            j = args.run_id
            print("Evaluating Model")
        else:
            print("Evaluating Model {}/{}".format(j+1, ensemble_folds))

        #checkpoint = torch.load(f=("models/best_"
        checkpoint = torch.load(f=("models/checkpoint_"
                                   "f-{}_r-{}_s-{}_t-{}"
                                   ".pth.tar").format(fold_id,
                                                      j,
                                                      args.seed,
                                                      args.sample),
                                map_location=args.device)
        model.load_state_dict(checkpoint["state_dict"])

        model.eval()
        idx, comp, pred, prec_embed, y_test, subset_accuracy, total = evaluate(generator=test_generator,
                                            model=model,
                                            criterion=criterion,
                                            optimizer=None,
                                            device=args.device,
                                            threshold=args.threshold,
                                            task="test")
        y_ensemble[j,:] = pred
        y_ensemble_prec_embed[j,:] = prec_embed


    y_pred = np.mean(y_ensemble, axis=0)
    y_prec_embed = np.mean(y_ensemble_prec_embed, axis=0)
    y_test = np.array(y_test)

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
    best_subset_acc = [{k:v} for k, v in subset_acc_dict.items() if v == max_acc]
    best_thresh = list(best_subset_acc[0].keys())[0]
    #print(best_thresh)
    best_logit_thresh = np.log(best_thresh / (1 - best_thresh))
    # get uncertainty estimate from ensemble for best_subset_acc threshold
    ensemble_accs = [accuracy_score(test_elems, pred_logits > best_logit_thresh) for pred_logits in y_ensemble]
    ensemble_error = np.std(ensemble_accs)

    print(f"Best Subset 0/1 score: {max_acc} +/- {ensemble_error} at {best_thresh}")
    print("Ensemble accuracies:", ensemble_accs)
    print("Subset 0/1 scores:", subset_acc_dict)
    print("Hamming Loss:", hamming_dict)
    print("F1 Score (weighted):", f1)

    print("y_pred", np.shape(y_pred))
    print("y_prec_embed", np.shape(y_prec_embed))
    print("y_test", np.shape(y_test))
    print("idx", np.shape(idx))

    # save results
    results = [y_pred, y_test, y_prec_embed, idx]
    with open("results/test_results_"
                                "f-{}_r-{}_s-{}_t-{}"
                                ".pkl".format(fold_id,
                                                   args.run_id,
                                                   args.seed,
                                                   args.sample), 'wb') as f:
        pkl.dump(results, f)
    print(f'Dumped logits, targets, prec_embeddings, and ids to results file')

if __name__ == "__main__":
    args = input_parser()

    print("The model will run on the {} device".format(args.device))

    main()