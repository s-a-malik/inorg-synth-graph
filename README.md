# inorg-synth-graph

Inorganic Reaction Representation Learning and Product Prediction

Cavendish Laboratory, University of Cambridge

## Dependancies 

See requirements.txt file

The raw dataset used for this work can be downloaded using the following command (linux):

```sh
wget -cO - https://ndownloader.figshare.com/files/17412674  > data/solid-state_dataset_2019-06-27.json
```

More recent versions of the dataset are released by the original authors [here](https://github.com/CederGroupHub/text-mined-synthesis_public)

The element embeddings used in this work are found here: [Unsupervised word embeddings capture latent knowledge from materials science literature](https://www.nature.com/articles/s41586-019-1335-8)

## Preprocess

dodgy_dois.txt contains the the dois of reactions incompatible with the model.
preprocess.py is used to generate the dataframes and supporting files from the raw data. 
This takes dodgy_dois.txt as an input, and the number of elements and precursors can be adjusted using optional arguments.

## Training and Testing

train.py is used for training the reaction graph model.
train_actions.py is used for training the reaction graph model with action sequences.
reaction_graph_actions/actions_rnn.py is used for training the action sequence autoencoder.
train_no_graph.py is used for training the non-graph baseline model.
train_stoich.py is used for training the stoichiometry prediction model.

The results from the product prediction model are then used in the stoichiometry prediction model.

Model dimensions and Hyperparameters are set using argparse flags.

## Example Usage

Preprocessing:
```
preprocess.py --dataset data/solid-state_dataset_2019-06-27.json \
              --dodgy-dois data/dodgy_dois.txt \
              --ps _10_precs \
              --max-prec 10 --min-prec 2
```

Training Action Autoencoder:
```
reaction_graph_actions/action_rnn.py --data-path data/datasets/dataset_10_precs.pkl \
                                     --action-path data/datasets/action_dict_10_precs.json 
```

Training product element prediction model (with actions):
```
train_actions.py --data-path data/datasets/dataset_10_precs.pkl \
                 --fea-path data/embeddings/magpie_embed_10_precs.json \
                 --action-path data/datasets/action_dict_10_precs.json \
                 --elem-path data/datasets/elem_dict_10_precs.json \
                 --action-rnn models/checkpoint_rnn_f-0_s-0_t-1.pth.tar \
                 --train-rnn --mask --amounts \
                 --ensemble 5 
```

Get reaction embeddings for full dataset (required for training stoichiometry prediction)
```
train_actions.py --data-path data/datasets/dataset_10_precs.pkl \
                 --fea-path data/embeddings/magpie_embed_10_precs.json \
                 --action-path data/datasets/action_dict_10_precs.json \
                 --elem-path data/datasets/elem_dict_10_precs.json \
                 --action-rnn models/checkpoint_rnn_f-0_s-0_t-1.pth.tar \
                 --train-rnn --mask --amounts \
                 --ensemble 5 \
                 --test-size 1 --evaluate
```

Training stoichiometry prediction model (using correct element predictions):
```
train_stoich.py --data-path results/test_results_f-0_r-0_s-0_t-1.pkl \
                 --elem-path data/datasets/elem_dict_10_precs.json \
                 --elem-fea-path data/embeddings/matscholar-embedding.json \
                 --use-correct-targets \
                 --ensemble 5 --fold-id 1
```

For end-to-end testing, run --evaluate on the trained product prediction model, then --evaluate on the trained stoichiometry prediction model (removing --use-correct-targets flag) using the product prediction results as an input file.

## Acknowledgements

The following work was used as a starting point for developing this work:
- Goodall, R.E., & Lee, A.A. (2019). https://arxiv.org/abs/1910.00617.

The dataset from the following was used in this work:
- Kononova, O., Huo, H., He, T. et al. (2019)  https://doi.org/11.1038/s41597-019-0224-1 




