import os
import sys
import argparse
import functools
from itertools import permutations

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
    parser = argparse.ArgumentParser(description="Inorganic Reaction Product Predictor, baseline model")

    # dataset inputs
    parser.add_argument("--data-path",
                        type=str,
                        default="data/datasets/dataset_prec3_df_all_2104.pkl",
                        metavar="PATH",
                        help="dataset path")
    parser.add_argument("--fea-path",
                        type=str,
                        default="data/embeddings/magpie_embed_prec3_df_all_2104.json",
                        metavar="PATH",
                        help="Precursor feature path")
    parser.add_argument('--elem-path',
	                    type=str,   
                        nargs='?',  
                        default='data/datasets/elem_dict_prec3_df_all_2104.json', 
	                    help="Path to element dictionary")
    parser.add_argument('--prec-type',
	                    type=str,   
                        nargs='?',  
                        default='stoich', 
	                    help="Type of input, stoich or magpie")
    parser.add_argument('--intermediate-dim',
                        type=int,   
                        nargs='?', 
                        default=128,
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
                        default=128,
                        type=int,
                        metavar="N",
                        help="mini-batch size")
    parser.add_argument("--val-size",
                        default=0.0,
                        type=float,
                        metavar="N",
                        help="proportion of data used for validation")
    parser.add_argument("--test-size",
                        default=0.2,
                        type=float,
                        metavar="N",
                        help="proportion of data used for testing")
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
                        default="BCE",
                        type=str,
                        metavar="str",
                        help="choose a Loss Function")
    parser.add_argument("--threshold",
                        default=0.5,
                        type=float,
                        metavar='prob',
                        help="Threshold for element presence in product (probability)")
    parser.add_argument("--reg-weight",
                        default=0,
                        type=float,
                        metavar="float",
                        help="Weight for regularisation loss")
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
        self.prec_type = prec_type
        self.augment = augment

        precs_stoich = np.array(df['prec_stoich'].tolist())
        precs_magpie = np.array(df['prec_magpie'].tolist())

        if prec_type == 'stoich':
            self.max_prec = precs_stoich.shape[1]
            self.embedd_dim = precs_stoich.shape[2]
        elif prec_type == 'magpie':
            self.max_prec = precs_magpie.shape[1]
            self.embedd_dim = precs_magpie.shape[2]           
        else:
            raise NameError("Only stoich or magpie allowed as --prec-type")

        targets = np.array(df['target'].tolist())

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
            _, prec_stoich, prec_magpie, _, _, _, target = self.df.iloc[idx]
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

        return (precs, prec_elems), \
            target, idx

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
    