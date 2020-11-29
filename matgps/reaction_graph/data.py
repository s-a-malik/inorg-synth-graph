import os
import json
import functools

import numpy as np
import pickle as pkl

import torch
from torch.utils.data import Dataset

from matgps.utils import LoadFeaturiser


class ReactionData(Dataset):
    """
    The ReactionData dataset is a wrapper for the dataset.
    """
    def __init__(self, data_path, fea_path, elem_dict_path, prec_type, amounts):
        """
        Inputs
        ----------
        data_path: string
            path to reaction data dataframe
        fea_path: string
            path to precursor feature dictionary
        elem_dict_path: string
            path to element dictionary
        prec_type: string
            type of embedding to use: magpie or stoich
        amounts: bool
            Whether to use molar amounts of precursors
        """

        # dataset
        assert os.path.exists(data_path), \
            "{} does not exist!".format(data_path)
        with open(data_path, 'rb') as f:
            #data = pkl.load(f)
            df = pkl.load(f)
        print(df)
        self.df = df

        # embeddings
        assert os.path.exists(fea_path), "{} does not exist!".format(fea_path)
        self.prec_features = LoadFeaturiser(fea_path)

        # elem_dict
        with open(elem_dict_path, 'r') as json_file:
            elem_dict = json.load(json_file)
        self.elem_dict = elem_dict

        self.amounts = amounts
        self.prec_type = prec_type
        if prec_type == 'magpie':
            self.prec_fea_dim = self.prec_features.embedding_size()
        elif prec_type == 'stoich':
            self.prec_fea_dim = len(df['prec_stoich'][0][0])

    def __len__(self):
        return len(self.df)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        """
        Parameters
        ----------
        M: number of precursors in reaction
        target_dim: Number of elements in dataset, dimension of target vector

        Returns
        -------
        prec_weights: torch.Tensor shape (M, 1)
            weights of precursors in the reaction
        prec_fea: torch.Tensor shape (M, n_fea)
            features of precursors in the reaction
        self_fea_idx: torch.Tensor shape (M*M, 1)
            list of self indicies
        nbr_fea_idx: torch.Tensor shape (M*M, 1)
            list of neighbour indicies
        target: torch.Tensor shape (target_dim, 1)
            target stoichiometry for reaction
        idx: torch.Tensor shape (1,)
            input id for the reaction
        """

        # get the materials and target for a particular reaction
        prec_stoich, materials, target = self.df.iloc[idx][["prec_stoich", "prec_roost_am", "target"]]
        # _, prec_stoich, _, _, materials, _, target = self.df.iloc[idx]
        precursors = [prec[0] for prec in materials]

        if self.amounts:
            # use precursor amounts as weighting
            weights = [prec[1] for prec in materials]
        else:
            # make weights for materials equal
            weights = [1 for prec in materials]

        weights = np.atleast_2d(weights).T / np.sum(weights)
        assert len(precursors) != 1, \
            "crystal {}: {}, is a pure system".format(idx, precursors)

        if self.prec_type == 'magpie':
            # get embeddings for materials - shape (num_materials, embedding)
            material_fea = np.vstack([self.prec_features.get_fea(prec)
                                    for prec in precursors])
        elif self.prec_type == 'stoich':
            # use stoich as element embeddings, ignoring empty ones
            material_fea = np.vstack([prec for prec in prec_stoich if not len(np.nonzero(prec)[0]) < 1])
        else:
            raise NameError("prec_type does not exist")

        env_idx = list(range(len(precursors)))
        self_fea_idx = []
        nbr_fea_idx = []
        for i, _ in enumerate(precursors):
            nbrs = precursors[:i]+precursors[i+1:]
            self_fea_idx += [i]*len(nbrs)
            nbr_fea_idx += env_idx[:i]+env_idx[i+1:]

        # get elements in precursors
        all_precs = torch.sum(torch.Tensor(prec_stoich), dim=0)
        # add other elements
        all_precs[self.elem_dict['O']] += 1
        #all_precs[self.elem_dict['H']] += 1
        prec_elems = torch.where(all_precs != 0, torch.ones_like(all_precs), all_precs)

        # convert all data to tensors
        material_weights = torch.Tensor(weights)
        material_fea = torch.Tensor(material_fea)
        self_fea_idx = torch.LongTensor(self_fea_idx)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        target = torch.Tensor(target)

        return (material_weights, material_fea, self_fea_idx, nbr_fea_idx, prec_elems), \
            target, materials, idx


def collate_batch(dataset_list):
    """
    Collate a list of data and return a batch for reaction graph inputs

    Parameters
    ----------
    n_i: number of precursors in reaction i
    N = sum(n_i); N0 = sum(i)
    prec_fea_len: dimension of precursor features


    Input
    -----

    dataset_list: list of tuples for each data point.
        (prec_weights, prec_fea, self_fea_idx, nbr_fea_idx, prec_elems),
            target, comp, reaction_id)

        prec_weights: torch.Tensor shape (n_i)
        prec_fea: torch.Tensor shape (n_i, prec_fea_len)
        self_fea_idx: torch.LongTensor shape (n_i*n_i)
        nbr_fea_idx: torch.LongTensor shape (n_i*n_i)
        target: torch.Tensor shape (target_dim)
        reaction_id: int

    Returns
    -------
    batch_prec_fea: torch.Tensor shape (N, orig_prec_fea_len)
        precursor features from precursor type
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
        Indices of M neighbors of each precursor
    reaction_prec_idx: list of torch.LongTensor of length N0
        Mapping from the reaction idx to prec idx
    batch_prec_elems: torch.Tensor shape (N0, stoich_dim)
        Mask for target output using only the elements present in the precursors
    batch_target: torch.Tensor shape (N0, stoich_dim)
        Target value for prediction
    batch_comp: list
        Elements in target
    batch_reaction_ids: torch.Tensor shape (N0,)
        input id for the reaction
    """
    # define the lists
    batch_prec_weights = []
    batch_prec_fea = []
    batch_self_fea_idx = []
    batch_nbr_fea_idx = []
    reaction_prec_idx = []
    batch_prec_elems = []
    batch_target = []
    batch_comp = []
    batch_reaction_ids = []

    cry_base_idx = 0
    for i, ((prec_weights, prec_fea, self_fea_idx, nbr_fea_idx, prec_elems),
            target, comp, reaction_id) in enumerate(dataset_list):
        # number of precursors for this reaction
        n_i = prec_fea.shape[0]

        # batch the features together
        batch_prec_weights.append(prec_weights)
        batch_prec_fea.append(prec_fea)

        # mappings from neighbours to precurosrs
        batch_self_fea_idx.append(self_fea_idx+cry_base_idx)
        batch_nbr_fea_idx.append(nbr_fea_idx+cry_base_idx)

        # mapping from precursors to reactions
        reaction_prec_idx.append(torch.tensor([i]*n_i))

        # batch precursor elements
        batch_prec_elems.append(prec_elems)

        # batch the targets and ids
        batch_target.append(target)
        batch_comp.append(comp)
        batch_reaction_ids.append(reaction_id)

        # increment the id counter
        cry_base_idx += n_i

    return (torch.cat(batch_prec_weights, dim=0),
            torch.cat(batch_prec_fea, dim=0),
            torch.cat(batch_self_fea_idx, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            torch.cat(reaction_prec_idx),
            torch.stack(batch_prec_elems, dim=0)), \
        torch.stack(batch_target, dim=0), \
        batch_comp, \
        batch_reaction_ids


if __name__ == "__main__":
    pass
