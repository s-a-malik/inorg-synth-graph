# inorg-synth-graph

Inorganic Reaction Representation Learning and Product Prediction.

Implementation of [Materials Graph Transformer predicts the outcomes of inorganic reactions with reliable uncertainties](https://arxiv.org/abs/2007.15752).

## Dependancies

See requirements.txt file

The raw dataset used for this work can be downloaded using the following command (linux):

```sh
mkdir data/datasets
wget -cO - https://ndownloader.figshare.com/files/17412674  > data/datasets/solid-state_dataset_2019-06-27.json
```

More recent versions of the dataset are released by the original authors [here](https://github.com/CederGroupHub/text-mined-synthesis_public)

The element embeddings used in this work are found here: [Unsupervised word embeddings capture latent knowledge from materials science literature](https://www.nature.com/articles/s41586-019-1335-8)

## Preprocess

preprocess.py is used to generate the dataframes and supporting files from the raw data. The number of elements and precursors can be adjusted using optional arguments.

dodgy_dois.txt contains the the dois of reactions incompatible with the model.


## Training and Testing

train.py is used for training the reaction graph model.

train_actions.py is used for training the reaction graph model with action sequences.

reaction_graph_actions/actions_rnn.py is used for training the action sequence autoencoder.

train_no_graph.py is used for training the non-graph baseline model.

train_stoich.py is used for training the stoichiometry prediction model.

Model dimensions and Hyperparameters are set using argparse flags.

## Example Usage

Preprocessing:
```sh
python preprocess.py --dataset data/datasets/solid-state_dataset_2019-06-27.json \
    --max-prec 10 --min-prec 2 \
    --ps _10_precs --seed 0
```

Training Action Autoencoder:
```sh
python train_action_rnn.py --train-path data/train_10_precs.pkl \
    --train-path data/test_10_precs.pkl \
    --action-path data/action_dict_10_precs.json
```

Training product element prediction model (with actions):
```sh
python train_reaction_graph_with_actions.py --train-path data/train_3_precs.pkl \
    --test-path data/test_10_precs.pkl \
    --fea-path data/magpie_embed_10_precs.json \
    --action-path data/action_dict_10_precs.json \
    --elem-path data/elem_dict_10_precs.json \
    --action-rnn models/checkpoint_rnn_f-0_s-0_t-1.pth.tar \
    --train-rnn --mask --amounts \
    --ensemble 5
```

Get reaction embeddings for full dataset (required for training stoichiometry prediction)
```sh
python train_reaction_graph_with_actions.py --train-path data/train_10_precs.pkl \
    --test-path data/test_10_precs.pkl \
    --fea-path data/magpie_embed_10_precs.json \
    --action-path data/action_dict_10_precs.json \
    --elem-path data/elem_dict_10_precs.json \
    --action-rnn models/checkpoint_rnn_f-0_s-0_t-1.pth.tar \
    --train-rnn --mask --amounts \
    --ensemble 5 \
    --get-reaction-emb
```

Training stoichiometry prediction model (using correct element predictions):
```sh
python train_stoich.py --train-path data/train_emb_f-1_r-0_s-0_t-1.pkl \
    --test-path data/test_emb_f-1_r-0_s-0_t-1.pkl \
    --elem-path data/elem_dict_10_precs.json \
    --elem-fea-path data/embeddings/matscholar-embedding.json \
    --use-correct-targets \
    --ensemble 5 --fold-id 1
```

For end-to-end testing, run --evaluate on the trained product prediction model, then --evaluate on the trained stoichiometry prediction model (removing --use-correct-targets flag) using the product prediction results as an input file.

## Acknowledgements

The following work was used as a starting point for developing this work:
- Goodall, R.E., & Lee, A.A. (2019). https://arxiv.org/abs/1910.00617.
