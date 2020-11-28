import os
import sys
import argparse
import functools

import pickle as pkl
import json

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset


class ProductData(Dataset):
    """
    The ProductData dataset is a wrapper for the product element prediction results dataset.
    """
    def __init__(self, data_path, fea_path, elem_path, threshold, use_correct_targets):
        """
        Inputs
        --------
        data_path: string
            Path to results file from product prediction
        fea_path: string
            Path to element feature vectors
        elem_path: string
            Path to element dictionary
        threshold: float
            Threshold probability for elemental presence
        use_correct_targets: bool
            Whether to use correct element predictions for training purposes
        """
        assert os.path.exists(data_path), \
            "{} does not exist!".format(data_path)
        with open(data_path, 'rb') as f:
            logits, targets, prec_embed, id = pkl.load(f)

        # save correct logits for training purposes
        if use_correct_targets:
            correct_logits = []
            for reaction in range(len(logits)):
                correct_logits.append([2.5 if elem != 0 else -999 for elem in targets[reaction]])
            df = pd.DataFrame(data={'logits': map(list, correct_logits), 'targets': map(list, targets),
                                    'prec_embed': map(list, prec_embed), 'id': id})
        else:
            df = pd.DataFrame(data={'logits': map(list, logits), 'targets': map(list, targets),
                                    'prec_embed': map(list, prec_embed), 'id': id})

        print(df)
        self.df = df

        assert os.path.exists(fea_path), "{} does not exist!".format(fea_path)
        assert os.path.exists(elem_path), "{} does not exist!".format(elem_path)
        # element embeddings
        self.atom_features = LoadFeaturiser(fea_path)
        self.atom_fea_dim = self.atom_features.embedding_size()
        self.reaction_fea_dim = len(df['prec_embed'].iloc[0])
        # index to element dictionary (reverse)
        with open(elem_path, 'r') as f:
            elem_dict = json.load(f)
        self.elem_dict = {v: k for k, v in elem_dict.items()}

        # convert threshold to logit threshold
        self.logit_threshold = np.log(threshold / (1 - threshold))

    def __len__(self):
        return len(self.df)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        """
        Returns
        -------
        atom_fea: torch.Tensor shape (M, n_fea)
            features of atoms in the product
        reaction_embed: torch.Tensor shape (reaction_fea,)
            reaction embedding
        target: torch.Tensor shape (target_dim,)
            target stoichiometries
        elements: list of strings
            Element symbols for elements in the product
        idx: torch.Tensor shape (1,)
            input id for the reaction
        """

        # get the elements, target and precursor embedding for a particular reaction
        logits, target, reaction_embed, reaction_id = self.df.iloc[idx]

        # get elemental presence
        elements = logits > self.logit_threshold
        elements = np.nonzero(elements)[0]
        # get element symbols
        elements = [self.elem_dict[index] for index in elements]

        # make weights for materials equal
        #weights = [1 for element in elements]
        # use logits as weighting
        # weights = [logits[index] for index in elements]
        # weights = np.atleast_2d(weights).T / np.sum(weights)

        # get non-zero stoich for target - in same order as elements
        target = [stoich for stoich in target if stoich]

        assert len(elements) != 1, \
            "crystal {}: {}, is a pure system".format(idx, elements)
        try:
            atom_fea = np.vstack([self.atom_features.get_fea(element)
                                  for element in elements])
        except AssertionError:
            print(elements)
            sys.exit()

        # convert all data to tensors
        atom_fea = torch.Tensor(atom_fea)
        target = torch.Tensor(target)
        reaction_embed = torch.Tensor(reaction_embed)

        return (atom_fea, reaction_embed), \
            target, elements, reaction_id


def collate_batch(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties. This means can collect different size batches

    Parameters
    ----------

    dataset_list: list of tuples for each data point.

      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      reaction_embed: torch.Tensor shape (reaction_fea_len,)
      target: torch.Tensor shape (target_dim,)
      elements: list shape (n_i)
      reaction_id: int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
        Atom features from atom type
    reaction_elem_idx: list of torch.LongTensor (N0,)
        Mapping from the crystal idx to atom idx
    batch_reaction_embed: torch.Tensor shape (N0, reaction_fea_len)
        Reaction embeddings
    target: torch.Tensor shape (N, target_dim)
        Target stoichs for prediction
    batch_reaction_ids: list (N0)
    """
    # define the lists
    batch_atom_fea = []
    reaction_elem_idx = []
    batch_reaction_embed = []
    batch_target = []
    batch_elements = []
    batch_reaction_ids = []

    reaction_base_idx = 0
    for i, ((atom_fea, reaction_embed),
            target, elements, reaction_id) in enumerate(dataset_list):

        # number of atoms for this reaction
        n_i = atom_fea.shape[0]

        # batch the features together
        batch_atom_fea.append(atom_fea)

        # mapping from atoms to reactions
        reaction_elem_idx.append(torch.tensor([i]*n_i))

        # batch reaction embedding
        batch_reaction_embed.append(reaction_embed)

        # batch the targets and ids
        batch_target.append(target)
        batch_elements.append(elements)
        batch_reaction_ids.append(reaction_id)

        # increment the id counter
        reaction_base_idx += n_i

    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(reaction_elem_idx, dim=0),
            torch.stack(batch_reaction_embed, dim=0)), \
        torch.cat(batch_target, dim=0), \
        batch_elements, \
        batch_reaction_ids


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Featuriser(object):
    """
    Base class for featurising nodes and edges.
    """
    def __init__(self, allowed_types):
        self.allowed_types = set(allowed_types)
        self._embedding = {}

    def get_fea(self, key):
        assert key in self.allowed_types, "{} is not an allowed material type".format(key)
        return self._embedding[key]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.allowed_types = set(self._embedding.keys())

    def get_state_dict(self):
        return self._embedding

    def embedding_size(self):
        return len(self._embedding[list(self._embedding.keys())[0]])

class LoadFeaturiser(Featuriser):
    """
    Initialize precursor feature vectors using a JSON file, which is a python
    dictionary mapping from material to a list representing the
    feature vector of the precursor.

    Parameters
    ----------
    embedding_file: str
        The path to the .json file
    """
    def __init__(self, embedding_file):
        with open(embedding_file) as f:
            embedding = json.load(f)
        allowed_types = set(embedding.keys())
        super(LoadFeaturiser, self).__init__(allowed_types)
        for key, value in embedding.items():
            self._embedding[key] = np.array(value, dtype=float)

if __name__ == "__main__":

    data_path = 'data/datasets/test_results_prec3_amounts_roost_with_embed.pkl'
    embedding_path = 'data/embeddings/matscholar-embedding.json'
    elemdict_path = 'data/datasets/elem_dict_prec3_df_all_2104.pkl'
    threshold = 0.9

    dataset = ProductData(data_path, embedding_path, elemdict_path, threshold, use_correct_targets=True)
    (atom_fea, prec_embed), \
            target, elements, cry_id = dataset.__getitem__(1)

    print(atom_fea, target, elements, prec_embed, cry_id)



