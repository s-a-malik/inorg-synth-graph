import json
import functools

import numpy as np
import pickle as pkl

from itertools import permutations

import torch
from torch.utils.data import Dataset


class ReactionData(Dataset):
    """
    The ReactionData dataset is a wrapper for the dataset.
    """
    def __init__(self, data_path, elem_dict_path, prec_type, augment):
        """
        Inputs
        ----------
        data_path: string
            path to reaction data dataframe
        elem_dict_path: string
            path to element dictionary
        prec_type: string
            type of embedding to use: magpie or stoich
        augment: bool
            Whether to augment dataset with rearrangements
        """

        with open(data_path, 'rb') as f:
            df = pkl.load(f)
        print(df)

        self.df = df
        self.n_prec = df["n_prec"].max()
        self.prec_type = prec_type
        self.augment = augment

        precs_stoich = np.array(df['prec_stoich'].tolist())[:, :self.n_prec, :]
        precs_magpie = np.array(df['prec_magpie'].tolist())[:, :self.n_prec, :]

        if prec_type == 'stoich':
            self.max_prec = precs_stoich.shape[1]
            self.embedd_dim = precs_stoich.shape[2]
        elif prec_type == 'magpie':
            self.max_prec = precs_magpie.shape[1]
            self.embedd_dim = precs_magpie.shape[2]
        else:
            raise NameError("Only stoich or magpie allowed as --prec-type")

        targets = np.array(df['target'].tolist())

        if self.augment:
            # augment data with rearrangements
            augmented_precs_stoich = []
            augmented_precs_magpie = []
            augmented_targets = []
            for i in range(len(precs_stoich)):
                perms_stoich = list(permutations(precs_stoich[i]))
                perms_magpie = list(permutations(precs_magpie[i]))
                for perm in range(len(perms_stoich)):
                    augmented_precs_stoich.append(perms_stoich[perm])
                    augmented_precs_magpie.append(perms_magpie[perm])
                    augmented_targets.append(targets[i])

            self.precs_stoich = np.array(augmented_precs_stoich)
            self.precs_magpie = np.array(augmented_precs_magpie)
            self.targets = np.array(augmented_targets)

        # elem_dict
        with open(elem_dict_path, 'r') as json_file:
            elem_dict = json.load(json_file)
        self.elem_dict = elem_dict

    def __len__(self):
        if self.augment:
            return len(self.precs_stoich)

        return len(self.df)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        """
        Returns
        -------
        precs: torch.Tensor shape (max_prec, embedding_dim)
            input vector
        prec_elems: torch.Tensor shape (target_dim,)
            mask for elements only in precursor
        target: torch.Tensor shape (target_dim,)
            target stoichiometry vector
        idx: torch.Tensor shape (1,)
            input id for the reaction
        """
        # cry_id, composition, target = self.id_prop_data[idx]
        # get the materials and target for a particular reaction
        if self.augment:
            prec_stoich = torch.Tensor(self.precs_stoich[idx])
            prec_magpie = torch.Tensor(self.precs_magpie[idx])
            target = torch.Tensor(self.targets[idx])
        else:
            prec_stoich, prec_magpie, target = self.df.iloc[idx][["prec_stoich", "prec_magpie", "target"]]
            prec_stoich = prec_stoich[:self.n_prec]
            prec_magpie = prec_magpie[:self.n_prec]
            target = torch.Tensor(target)

        # get inputs
        if self.prec_type == 'stoich':
            precs = torch.Tensor(prec_stoich)
        elif self.prec_type == 'magpie':
            precs = torch.Tensor(prec_magpie)

        # get mask
        all_precs = torch.sum(torch.Tensor(prec_stoich), dim=0)
        # add oxygen
        all_precs[self.elem_dict['O']] += 1
        prec_elems = torch.where(all_precs != 0, torch.ones_like(all_precs), all_precs)

        return (precs, prec_elems), target, idx


if __name__ == "__main__":

    #data_path = 'data/datasets/dataset_prec3_amounts_roost.pkl'
    data_path = 'data/datasets/datasetdf_test.pkl'
    embedding_path = 'data/embeddings/magpie_embeddf_test.json'
    elem_dict_path = 'data/datasets/elem_dictdf_test.json'
    dataset = ReactionData(data_path, elem_dict_path, 'magpie', True)

    for i in range(6):
        (precs, prec_elems), \
                target, cry_id = dataset.__getitem__(i)
        print(precs.shape, prec_elems.shape, target.shape, cry_id)
    print(len(dataset))
