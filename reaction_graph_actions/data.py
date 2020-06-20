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
from torch.nn.utils.rnn import pad_sequence


def input_parser():
    """
    parse input
    """
    parser = argparse.ArgumentParser(description="Inorganic Reaction Product Predictor," 
                                                "reaction graph model with actions")

    # dataset inputs
    parser.add_argument("--data-path",
                        type=str,
                        default="data/datasets/dataset_10_precs.pkl",
                        metavar="PATH",
                        help="dataset path")
    parser.add_argument("--fea-path",
                        type=str,
                        default="data/embeddings/magpie_embed_10_precs.json",
                        metavar="PATH",
                        help="Precursor feature path")
    parser.add_argument('--action-rnn',
	                    type=str,   
                        nargs='?',  
                        default='models/checkpoint_rnn_f-1_s-0_t-1.pth.tar', 
	                    help="Path to trained action autoencoder")
    parser.add_argument('--action-path',
	                    type=str,   
                        nargs='?',  
                        default='data/datasets/action_dict_10_precs.json', 
	                    help="Path to action dictionary")
    parser.add_argument('--elem-path',
	                    type=str,   
                        nargs='?',  
                        default='data/datasets/elem_dict_10_precs.json', 
	                    help="Path to element dictionary")
    parser.add_argument('--prec-type',
	                    type=str,   
                        nargs='?',  
                        default='magpie', 
	                    help="Type of input, stoich or magpie")
    parser.add_argument('--latent-dim',
                        type=int,   
                        nargs='?', 
                        default=32,
                        help='Latent dimension for RNN hidden state')    
    parser.add_argument('--intermediate-dim',
                        type=int,   
                        nargs='?', 
                        default=256,
                        help='Intermediate model dimension')
    parser.add_argument('--target-dim',
                        type=int,   
                        nargs='?', 
                        default=81,
                        help='Target vector dimension')
    parser.add_argument("--prec-fea-len",
                        default=128,
                        type=int,
                        metavar="N",
                        help="Dimension of node features")
    parser.add_argument("--n-graph",
                        default=5,
                        type=int,
                        metavar="N",
                        help="number of graph layers")
    parser.add_argument('--mask',
                        action="store_true",
                        default=False,
                        help="Whether to mask output with precursor elements or not")  
    parser.add_argument('--amounts',
                        action="store_true",
                        default=False,
                        help="use precursor amounts as weights")


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
                        help="mini-batch size (default: 256)")
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
                        default=60,
                        type=int,
                        metavar="N",
                        help="number of total epochs to run")
    parser.add_argument("--loss",
                        default="BCE",
                        type=str,
                        metavar="str",
                        help="choose a Loss Function")
    parser.add_argument("--threshold",
                        default=0.9,
                        type=float,
                        metavar='prob',
                        help="Threshold for element presence in product (probability)")
    parser.add_argument("--reg-weight",
                        default=0,
                        type=float,
                        metavar="float",
                        help="Weight for regularisation loss")
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
                        default=1,
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
    parser.add_argument('--train-rnn',
                        action="store_true",
                        default=False,
                        help="Train rnn for elem prediction as well")   
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
    def __init__(self, data_path, fea_path, action_dict_path, elem_dict_path, prec_type, amounts):
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
        
        # actions
        self.actions = df['actions'].tolist()
        # dictionary of action sequences        
        assert os.path.exists(action_dict_path), \
            "{} does not exist!".format(action_dict_path)
        with open(action_dict_path, 'r') as json_file:
            action_dict = json.load(json_file)       
        action_dict_aug = {k:v+3 for k,v in action_dict.items()}
        action_dict_aug['<SOS>'] = 1
        action_dict_aug['<EOS>'] = 2
        action_dict_aug['<PAD>'] = 0
        self.action_dict = action_dict_aug
        print(action_dict_aug)
        self.action_fea_dim = len(action_dict_aug)

    def __len__(self):
        return len(self.df)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        """
        Parameters
        -------
        M: no. of elements in reaction
        n_fea: material embedding feature size
        stoich_dim: stoich of target feature size (no. of elements)
        A: no. of actions in reaction (including EOS)
        a_fea: action embedding feature size (including SOS, EOS, pad tokens)

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
        actions: torch.Tensor shape (A, a_fea)
            features of actions in reaction with additional token
        target: torch.Tensor shape (target_dim, 1)
            target stoichiometry for reaction
        materials: list of tuples variable length
            raw precursor strings
        idx: torch.Tensor shape (1,)
            input id for the reaction
        """
        # get the materials and target for a particular reaction
        _, prec_stoich, _, _, materials, actions_raw, target = self.df.iloc[idx]
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
            # get embeddings for materials
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
        actions = torch.Tensor(actions)
        target = torch.Tensor(target)

        return (material_weights, material_fea, self_fea_idx, nbr_fea_idx, actions, prec_elems), \
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
        actions: torch.Tensor shape (a_i, a_fea)
        target: torch.Tensor shape (target_dim)
        reaction_id: int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)
    A = sum(a_i) (padded)

    batch_prec_fea: torch.Tensor shape (N, orig_prec_fea_len)
        precursor features from precursor type
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
        Indices of M neighbors of each precursor
    reaction_prec_idx: list of torch.LongTensor of length N0
        Mapping from the reaction idx to prec idx
    batch_actions: torch.Tensor shape (N0, max(a_i), a_fea)
        Action features padded
    batch_actions_len: torch.Tensor shape (N0)
        length of each action sequence in batch
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
    batch_actions = []
    batch_actions_len = []
    batch_prec_elems = []
    batch_target = []
    batch_comp = []
    batch_reaction_ids = []

    cry_base_idx = 0
    for i, ((prec_weights, prec_fea, self_fea_idx, nbr_fea_idx, actions, prec_elems),
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

        # batch actions
        batch_actions.append(actions)
        batch_actions_len.append(len(actions))

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
            pad_sequence(batch_actions, batch_first=True, padding_value=0),
            torch.tensor(batch_actions_len),
            torch.stack(batch_prec_elems, dim=0)), \
        torch.stack(batch_target, dim=0), \
        batch_comp, \
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
    
    data_path = 'data/datasets/dataset_prec10_df_all_2104_prec3_dict.pkl'
    embedding_path = 'data/embeddings/magpie_embed_prec10_df_all_2104.json'
    elem_path = 'data/datasets/elem_dict_prec3_df_all_2104.json'
    action_path ='data/datasets/action_dict_prec3_df_all_2104.json'
    dataset = ReactionData(data_path, embedding_path, action_path, elem_path, prec_type='magpie', amounts=True)
    print(dataset.elem_dict) 
    (material_weights, material_fea, self_fea_idx, nbr_fea_idx, actions, prec_elems), \
            target, materials, cry_id = dataset.__getitem__(16064)

    print(material_weights, material_fea, self_fea_idx, nbr_fea_idx, actions, prec_elems, target, materials, cry_id)
    print(dataset.df.iloc[16064])
    print(len(dataset))
    # print(dataset.df.loc[dataset.df['dois'] == '10.1016/j.mseb.2003.09.034'])
