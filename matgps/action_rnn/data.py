import os
import json
import pickle as pkl

import torch
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn_utils


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
        action_dict_aug = {k: v+3 for k, v in action_dict.items()}
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
