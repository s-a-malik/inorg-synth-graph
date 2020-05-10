# inorg-synth-graph

Inorganic Reaction Representation Learning and Product Prediction
Part III Physics Project, University of Cambridge

## Dependancies 

See requirements.txt file

The raw dataset, embeddings and processed data are given in the data folder

## Preprocess

dodgy_dois.txt contains the the dois of reactions incompatible with the model.
preprocess.py is used to generate the dataframes and supporting files from the raw data. This takes dodgy_dois.txt as an input, and the number of elements and precursors can be adjusted using optional arguments.
Use the --df flag to save as a pandas dataframe (format used in the model)

## Training and Testing

train.py is used for training the reaction graph model
train_actions.py is used for training the reaction graph model with action sequences
reaction_graph_actions/actions_rnn.py is used for training the action sequence autoencoder
train_no_graph.py is used for training the non-graph baseline model
train_stoich.py is used for training the stoichiometry prediction model


The results from the product prediction model are then used in the stoichiometry prediction model

Model dimensions and Hyperparameters are set using argparse flags.

Use the --evaluate argument to evaluate the models


## Example Usage

Preprocessing:
```
preprocess.py --prec_preprocess df \
                        --dataset data/solid-state_dataset_2019-06-27_upd.json \
                        --elem-dict data/datasets/elem_dict \
                        --action-dict data/datasets/action_dict \
                        --clean-set data/datasets/dataset \
                        --magpie-embed data/embeddings/magpie_embed \
                        --ps _prec10_df_30_2104 \
                        --dodgy-dois data/dodgy_dois.txt \
                        --max-prec 10 --min_prec 2 \
                        --num-elem 30
```

Training Action Autoencoder:
```
reaction_graph_actions/action_rnn.py --data-path data/datasets/dataset_prec10_df_all_2104.pkl \
                             --action-path data/datasets/action_dict_prec10_df_all_2104.json \
                             --val-size 0 --test-size 0.2 --seed 0 \
                             --latent-dim 32 --num-layers 1 \
                             --optim SGD --lr 0.3 --loss CrossEntropy \
                             --epochs 70 --batch-size 128 \
                             --fold-id 260498132 \
```

Training product prediction model (with actions):
```
train_actions.py --data-path data/datasets/dataset_prec10_df_all_2104_prec3_dict.pkl \
                 --fea-path data/embeddings/magpie_embed_prec10_df_all_2104.json \
                 --action-path data/datasets/action_dict_prec10_df_all_2104.json \
                 --elem-path data/datasets/elem_dict_prec3_df_all_2104.json \
                 --action-rnn models/rnn_f-260498132.pth.tar \
                 --val-size 0 --test-size 0.2 --seed 0 \
                 --ensemble 5 --run-id 0 --fold-id 26049811 \
                 --atom-fea-len 128 --n-graph 5 \
                 --latent-dim 32 \
                 --intermediate-dim 256 --target-dim 81 \
                 --optim Adam --lr 0.0001 --loss BCE \
                 --epochs 60 --batch-size 256 --reg-weight 0 \
                 --train-rnn --mask --amounts --prec-type magpie \
```

Training stoichiometry prediction model:
```
train_stoich.py --data-path results/correct_prec10_rnn_26049811.pkl \
                 --elem-path data/datasets/elem_dict_prec10_df_all_2104.json \
                 --elem-fea-path data/embeddings/matscholar-embedding.json \
                 --seed 0 --sample 1 \
                 --val-size 0 --test-size 0.2 \
                 --ensemble 5 --run-id 0 --fold-id 26049812 \
                 --intermediate-dim 256 --n-heads 5 \
                 --threshold 0.5 \
                 --optim Adam --lr 0.0001 --loss MSE \
                 --epochs 200 --batch-size 256 
```

For end-to-end testing, run --evaluate on the trained product prediction model, then --evaluate on the trained stoichiometry prediction model using the product prediction results as an input file.

## Acknowledgements

The following work was used as a starting point for developing this work:
- Goodall, R.E., & Lee, A.A. (2019). https://arxiv.org/abs/1910.00617.

The dataset from the following was used in this work:
- Kononova, O., Huo, H., He, T. et al. (2019)  https://doi.org/11.1038/s41597-019-0224-1 




