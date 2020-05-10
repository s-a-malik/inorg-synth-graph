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


def input_parser():
    """
    parse input
    """
    parser = argparse.ArgumentParser(description="Inorganic Reaction Product Predictor," 
                                                "Stoichiometry prediction")
    # dataset inputs
    parser.add_argument("--data-path",
                        type=str,
                        default="results/correct_prec10_rnn_26049811.pkl",
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
                        default='data/datasets/elem_dict_prec10_df_all_2104.json', 
	                    help="Path to element dictionary")
    parser.add_argument('--intermediate-dim',
                        type=int,   
                        nargs='?', 
                        default=128,
                        help='Intermediate model dimension')
    parser.add_argument("--n-heads",
                        default=3,
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

    # optimiser inputs
    parser.add_argument("--epochs",
                        default=300,
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
                        default="SGD",
                        type=str,
                        metavar="str",
                        help="choose an optimizer; SGD, Adam or AdamW")
    parser.add_argument("--learning-rate", "--lr",
                        default=5e-4,
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


class ProductData(Dataset):
    """
    The ProductData dataset is a wrapper for the product element prediction results dataset.
    """
    def __init__(self, data_path, fea_path, elem_path, threshold):
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
        """
        assert os.path.exists(data_path), \
            "{} does not exist!".format(data_path)
        with open(data_path, 'rb') as f:
            raw_data = pkl.load(f)
        data = [map(list, x) for x in raw_data[:-1]]
        data.append(raw_data[-1])
        df = pd.DataFrame(data={'logits': data[0], 'targets': data[1], 
                                'prec_embed': data[2], 'id': data[3]})

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
    
    data_path = 'data/datasets/test_results_prec3_amounts_roost_with_embed_correct.pkl'
    embedding_path = 'data/embeddings/onehot-embedding.json'
    elemdict_path = 'data/datasets/elem_dict_prec3_amounts_roost.pkl'
    threshold = 0.9
    
    dataset = ProductData(data_path, embedding_path, elemdict_path, threshold)
    (atom_fea, prec_embed), \
            target, elements, cry_id = dataset.__getitem__(1)

    print(atom_fea, target, elements, prec_embed, cry_id)

        

