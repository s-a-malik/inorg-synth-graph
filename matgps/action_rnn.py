"""Module containing RNN autoencoder for action sequences
and its training routines: preprocessing sequences, training, saving
"""

import os
import json
import random
import shutil
import datetime
import pickle as pkl

from tqdm.autonotebook import trange
from sklearn.model_selection import train_test_split as split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from matgps.utils import AverageMeter, plot_grad_flow


class ActionData(Dataset):
    """
    The ActionData dataset is a wrapper for a dataset for action sequences.
    """
    def __init__(self, data_path, action_dict_path):
        """
        Inputs:
        data_path - path to reaction dataframe
        action_dict_path - path to action dictionary
        """
        assert os.path.exists(data_path), \
            "{} does not exist!".format(data_path)

        with open(data_path, 'rb') as f:
            df = pkl.load(f)
        # print(df)

        with open(action_dict_path, 'r') as json_file:
            action_dict = json.load(json_file)

        self.df = df
        self.actions = df['actions'].tolist()

        # dictionary of action sequences
        action_dict_aug = {k:v+3 for k,v in action_dict.items()}
        action_dict_aug['<SOS>'] = 1
        action_dict_aug['<EOS>'] = 2
        action_dict_aug['<PAD>'] = 0
        self.action_dict = action_dict_aug
        self.action_fea_dim = len(action_dict_aug)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns
        -------
        actions: torch.Tensor shape (A, a_fea)
            features of actions in reaction
            augmented with SOS and EOS tokens
        """

        actions_raw = self.df.iloc[idx]["actions"]

        # action sequences in tensor form: OHE vectors unpadded
        actions = torch.Tensor(actions_raw)

        # add extra feature dimensions for pad and EOS
        if len(actions) != 0:
            actions = torch.cat((torch.zeros(len(actions), 3), actions), dim=1)

        # add EOS/SOS to sequence
        eos = torch.zeros(1, self.action_fea_dim)
        sos = torch.zeros(1, self.action_fea_dim)
        sos[0, 1] = 1
        eos[0, 2] = 1
        actions = torch.cat((sos, actions, eos), dim=0)

        return actions


def collate_batch(batch):
    """Collate Fn for dataloader.
    """
    # pad sequences
    batch_padded = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=0)
    batch_lens = torch.Tensor([len(x) for x in batch])

    return (batch_padded, batch_lens)


class LSTM(nn.Module):
    """LSTM autoencoder for action sequences
    init: input_dim: action embedding size
    latent_dim: latent rep size
    num_layers: number of LSTM layers
    """
    def __init__(self, input_dim, latent_dim, device, num_layers=1, embedding_dim=8):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.device = device

        self.fc_in = nn.Linear(self.input_dim, self.embedding_dim)
        self.dropout = nn.Dropout(p=0.2)
        self.encoder = nn.LSTM(self.embedding_dim, self.latent_dim, self.num_layers, batch_first=True)

        self.fc_in_decode = nn.Linear(self.input_dim, self.embedding_dim)
        self.dropout_decode = nn.Dropout(p=0.2)
        self.decoder = nn.LSTM(self.embedding_dim, self.latent_dim, self.num_layers, batch_first=True)
        self.fc_out = nn.Linear(self.latent_dim, self.input_dim)

    def forward(self, x, x_lens, teacher_forcing_ratio=0):
        """x: Tensor (batch_size, max length seq, feature dim)
        x_lens: list (batch size)
        teacher_forcing_ratio: float range 0 - 1
        """

        sequence_len = x_lens  # sequence len for each input in batch

        # embed inputs (linear layer only applies transformation to last dim)
        x_embedded = self.dropout(self.fc_in(x))

        # pack the padded sequences (gets rid of padded ones)
        x_packed = rnn_utils.pack_padded_sequence(x_embedded, sequence_len,
                                                    batch_first=True, enforce_sorted=False).to(self.device)

        # encode
        _, last_hidden = self.encoder(x_packed)
        encoded = last_hidden[0].view(x.shape[0], self.num_layers, -1)
        # take only last layer hidden state for input
        encoded = encoded[:, -1, :]

        # decode
        y = LSTM._decode(self.decoder, self.fc_in_decode, self.dropout_decode,
                         self.fc_out, x, last_hidden, x.shape[1], teacher_forcing_ratio, self.device)

        return y, x_lens, encoded

    @staticmethod
    def _decode(
        decoder: nn.LSTM,
        fc_in: nn.Linear,
        dropout: nn.Dropout,
        output_layer: nn.Linear,
        input_sequence: torch.Tensor,
        hidden: (torch.Tensor, torch.Tensor),
        steps: int,
        teacher_forcing_ratio: float,
        device
    ):
        """Decode from latent rep, using teacher forcing
        """

        # initialise output (same dim as input - padded)
        outputs = torch.zeros_like(input_sequence).to(device)
        # first input to the decoder is the <sos> tokens
        # use all batches, first token in each batch
        input_ = input_sequence[:, 0].view(input_sequence.shape[0], 1, -1).to(device)
        state = hidden

        for t in range(1, steps):

            input_embedded = dropout(fc_in(input_))
            output, state = decoder(input_embedded, state)
            output = output_layer(output)

            # place predictions in a tensor holding predictions for each token
            outputs[:, t] = output.squeeze(dim=1)

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our prediction
            top1 = output.argmax(dim=2).view(input_.shape[0], 1)
            top1 = F.one_hot(top1, num_classes=input_.shape[-1]).float()

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input_ = input_sequence[:, t].view(input_sequence.shape[0], 1, -1).to(device) if teacher_force else top1.to(device)

        return outputs


def custom_loss(output, target):
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


def init_optim(model):

    # Select Loss Function
    if args.loss == "CrossEntropy":
        criterion = custom_loss
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

    return criterion, optimizer

def evaluate(generator, model, criterion, optimizer, device, task="train", verbose=False, teacher_forcing=False):
    """
    evaluate the model
    """

    if task == "test":
        model.eval()
        test_targets = []
        test_pred = []
        test_lens = []
        test_encoded = []
        test_total = 0
    else:
        loss_meter = AverageMeter()
        if task == "val":
            model.eval()
        elif task == "train":
            model.train()
        else:
            raise NameError("Only train, val or test is allowed as task")

    with trange(len(generator), disable=(not verbose)) as t:
        for (x_padded, x_lens) in generator:

            x_padded = x_padded.to(device)
            x_lens = x_lens.to(device)

            if teacher_forcing:
                output, output_lens, encoded = model(x_padded, x_lens, teacher_forcing_ratio=0.5)
            else:
                output, output_lens, encoded = model(x_padded, x_lens, teacher_forcing_ratio=0)

            if task == "test":
                # collect the model outputs
                test_targets += x_padded[:, 1:].tolist()
                test_pred += output[:, 1:].tolist()
                test_lens += output_lens.tolist()
                test_encoded += encoded.tolist()
                test_total += output.size(0)
            else:
                # remove SOS token
                output = output[:, 1:, :].contiguous()
                # flatten
                output = output.view(-1, output.shape[-1])
                x_padded = x_padded[:, 1:, :].contiguous()
                x_padded = x_padded.view(-1, x_padded.shape[-1]).to(device)
                # get true class value for loss
                _, x_padded = x_padded.max(dim=1)

                loss = criterion(output, x_padded)
                loss_meter.update(loss.data.cpu().item(), x_padded.size(0))

                if task == "train":
                    # compute gradient and do SGD step
                    optimizer.zero_grad()
                    loss.backward()

                    # plot_grad_flow(model.named_parameters())

                    optimizer.step()

            t.update()

    if task == "test":
        return test_pred, test_lens, test_encoded, test_targets, test_total
    else:
        return loss_meter.avg


if __name__ == "__main__":

    args = input_parser()

    # gets raw action sequences (OHE embedded already)
    dataset = ActionData(args.data_path, args.action_path)
    print(dataset.action_dict)

    # get train/val/test generators - these form the padded sequences
    indices = list(range(len(dataset)))
    train_idx, test_idx = split(indices, random_state=args.seed,
                                test_size=args.test_size)
    train_set = torch.utils.data.Subset(dataset, train_idx[0::args.sample])
    test_set = torch.utils.data.Subset(dataset, test_idx)
    print("Shape of train, test set: ", train_set.__len__(), test_set.__len__())

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
        indices = list(range(len(train_set)))
        train_idx, val_idx = split(indices, random_state=args.seed,
                                   test_size=args.val_size/(1-args.test_size))
        train_subset = torch.utils.data.Subset(train_set, train_idx)
        val_subset = torch.utils.data.Subset(train_set, val_idx)
        print("Shape of train, val subset: ", train_subset.__len__(), val_subset.__len__())

    train_generator = DataLoader(train_subset, **params)
    val_generator = DataLoader(val_subset, **params)

    # initialise model and optimization
    model = LSTM(input_dim=dataset.action_fea_dim, latent_dim=args.latent_dim, device=args.device,
                    num_layers=args.num_layers, embedding_dim=args.embedding_dim)
    model.to(args.device)
    print(model)
    criterion, optimizer = init_optim(model)
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total Number of Trainable Parameters: {}".format(num_param))

    # try except structure used to allow keyboard interupts to stop training
    # without breaking the code
    start_epoch = 0
    if not args.evaluate:

        if args.lr_search:

            lr_finder = LRFinder(model, optimizer, criterion,
                                 metric="mse", device=args.device)
            lr_finder.range_test(train_generator, end_lr=1,
                                 num_iter=100, step_mode="exp")
            lr_finder.plot()
            exit()

        writer = SummaryWriter(log_dir=("runs/rnn-f-{f}_s-{s}_t-{t}_"
                                            "{date:%d-%m-%Y_%H:%M:%S}").format(
                                                date=datetime.datetime.now(),
                                                f=args.fold_id,
                                                s=args.seed,
                                                t=args.sample))

        checkpoint_file = ("models/checkpoint_rnn_"
                        "f-{}_s-{}_t-{}.pth.tar").format(args.fold_id,
                                                                args.seed,
                                                                args.sample)
        best_file = ("models/best_rnn_"
                    "f-{}_s-{}_t-{}.pth.tar").format(args.fold_id,
                                                       args.seed,
                                                       args.sample)
        best_loss = evaluate(generator=val_generator,
                                                    model=model,
                                                    criterion=criterion,
                                                    optimizer=None,
                                                    device=args.device,
                                                    task="val")

        try:
            for epoch in range(start_epoch, start_epoch+args.epochs):
                # Training
                t_loss = evaluate(generator=train_generator,
                                                model=model,
                                                criterion=criterion,
                                                optimizer=optimizer,
                                                device=args.device,
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
                                                        task="val")

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

                checkpoint_dict = {"epoch": epoch,
                                "state_dict": model.state_dict(),
                                "best_error": best_loss,
                                "optimizer": optimizer.state_dict(),
                                "args": vars(args)}
                torch.save(checkpoint_dict, checkpoint_file)
                if is_best:
                    shutil.copyfile(checkpoint_file, best_file)

                writer.add_scalar("loss/train", t_loss, epoch+1)
                writer.add_scalar("loss/validation", val_loss, epoch+1)


        except KeyboardInterrupt:
            pass

    # test set
    test_model(test_set, model)
