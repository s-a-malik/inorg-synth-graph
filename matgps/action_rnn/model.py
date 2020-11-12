"""Module containing RNN autoencoder for action sequences
and its training routines: preprocessing sequences, training, saving
"""


import random

from tqdm.autonotebook import trange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

from matgps.utils import AverageMeter
from typing import Tuple


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
        hidden: Tuple[torch.Tensor, torch.Tensor],
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

    def evaluate(self, generator, criterion, optimizer, device, task="train", verbose=False, teacher_forcing=False):
        """
        evaluate the model
        """

        if task == "test":
            self.eval()
            test_targets = []
            test_pred = []
            test_lens = []
            test_encoded = []
            test_total = 0
        else:
            loss_meter = AverageMeter()
            if task == "val":
                self.eval()
            elif task == "train":
                self.train()
            else:
                raise NameError("Only train, val or test is allowed as task")

        with trange(len(generator), disable=(not verbose)) as t:
            for (x_padded, x_lens) in generator:

                x_padded = x_padded.to(device)
                x_lens = x_lens.to(device)

                if teacher_forcing:
                    output, output_lens, encoded = self(x_padded, x_lens, teacher_forcing_ratio=0.5)
                else:
                    output, output_lens, encoded = self(x_padded, x_lens, teacher_forcing_ratio=0)

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
    pass
