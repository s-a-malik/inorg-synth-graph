"""Preprocess Dataset for forward synthesis model
Define and save dictionaries for elements, actions, and magpie embeddings
"""


import os
import json
import argparse
import numpy as np
import pandas as pd
import pickle as pkl
from itertools import permutations

from sklearn.model_selection import train_test_split as split

from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers import composition as cf
from matminer.featurizers.conversions import StrToComposition

from data.exclude import BAD_DOI, BAD_PRECURSORS, BAD_TARGETS


def parse_args():
    """read arguments from command line
    """

    parser = argparse.ArgumentParser(
        description='preprocess.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset',
                        type=str,
                        nargs='?',
                        default='data/datasets/solid-state_dataset_2019-09-27_upd.json',
                        help="Path to dataset to use")

    parser.add_argument('--elem-dict',
                        type=str,
                        nargs='?',
                        default='data/elem_dict',
                        help="Path to element to index dictionary without extension")

    parser.add_argument('--action-dict',
                        type=str,
                        nargs='?',
                        default='data/action_dict',
                        help="Path to element to index dictionary without extension")

    parser.add_argument('--magpie-embed',
                        type=str,
                        nargs='?',
                        default='data/magpie_embed',
                        help="Path to magpie embeddings dictionary without extension")

    parser.add_argument('--clean-set',
                        type=str,
                        nargs='?',
                        default='data/dataset',
                        help="Path to full clean dataset to use without extension")

    parser.add_argument('--train-set',
                        type=str,
                        nargs='?',
                        default='data/train',
                        help="Path to train dataset to use without extension")

    parser.add_argument('--test-set',
                        type=str,
                        nargs='?',
                        default='data/test',
                        help="Path to test dataset to use without extension")

    parser.add_argument('--val-set',
                        type=str,
                        nargs='?',
                        default='data/val',
                        help="Path to val dataset to use without extension")

    parser.add_argument('--test-size',
                        type=float,
                        nargs='?',
                        default=0.2,
                        help="size of clean dataset for testing")

    parser.add_argument('--val-size',
                        type=float,
                        nargs='?',
                        default=0,
                        help="size of clean dataset for validation")

    parser.add_argument('--seed',
                        type=int,
                        nargs='?',
                        default=0,
                        help="Random seed for splitting data")

    parser.add_argument('--ps',
                        type=str,
                        nargs='?',
                        default='',
                        help="postscript on path for save files")

    parser.add_argument('--max-prec',
                        type=int,
                        nargs='?',
                        default=10,
                        help='Max number of precursors per reaction.')

    parser.add_argument('--min-prec',
                        type=int,
                        nargs='?',
                        default=2,
                        help='Min number of precursors per reaction. Default 2')

    parser.add_argument('--augment',
                        action="store_true",
                        help="augment data with precursor rearrangements")

    parser.add_argument('--split-prec-amts',
                        action="store_true",
                        help="split out data for the baseline model")

    parser.add_argument('--num-elem',
                        type=int,
                        metavar='N',
                        nargs='?',
                        default=-1,
                        help='Take N most common elements only. Default: -1 (all)')

    args = parser.parse_args()

    return args


def load_dataset():
    """Load dataset from path listed in arguments.
    Uses exclude.py file to selectively remove troublesome reactions
    Returns trimmed dataset with only reactions with
    args.min_prec <= no. of precursors <= args.max_prec
    """
    # Load raw data
    with open(args.dataset, 'r') as json_file:
        raw_data = json.load(json_file)

    print('Untrimmed dataset size', len(raw_data))
    data = []

    for reaction in range(len(raw_data)):
        # check if number of precursors is within limits and no dodgy precursors
        if (
            len(raw_data[reaction]['precursors']) > args.max_prec or
            len(raw_data[reaction]['precursors']) < args.min_prec or
            raw_data[reaction]['doi'] in BAD_DOI or
            any(raw_data[reaction]['targets_string'][x]
                in BAD_TARGETS for x in range(len(raw_data[reaction]['targets_string']))
                ) or
            any(raw_data[reaction]['precursors'][x]['material_formula']
                in BAD_PRECURSORS for x in range(len(raw_data[reaction]['precursors']))
                )
        ):
            pass
        else:
            data.append(raw_data[reaction])

    print(f'Trimmed dataset size with {args.max_prec}:', len(data))

    return data


def normalise(stoich):
    """normalise a stoichiometry vector to composition
    i.e. components sum to 1
    """
    sum_stoich = np.sum(stoich)
    if sum_stoich != 0:
        stoich = stoich / sum_stoich

    return stoich


def decode_material(encoded_material, vec_to_elem_dict):
    """Input: -  encoded_material: vector
    - vec_to_elem_dict: dict mapping elements to indexes
    Returns: - decoded material (dictionary with stoich coeffs)
    """
    decoded_material = {vec_to_elem_dict[index]: val for index, val in enumerate(encoded_material)}

    return decoded_material


def dict_to_formula(material_dict):
    """Input a material in element dict form
    Returns formula of element (arbitrary order)
    """
    formula = ""
    for element, val in material_dict.items():
        if val != 0:
            formula = formula + f'{element}{val}'

    return formula


def find_elem_dict(data):
    """Returns a dict for mapping elements to values to encode materials
    by their stoichiometry
    """

    all_elements = []

    for reaction in range(len(data)):
        """
        # elements from precursors
        for precursor in range(len(data[reaction]['precursors'])):
            composition = data[reaction]['precursors'][precursor]['composition']
            elements = [composition[x]['elements'].keys() for x in range(len(composition))]
            elements = [item for t in elements for item in t]
            all_elements = all_elements + elements
        """
        # elements from targets
        composition = data[reaction]['target']['composition']
        # Remove non-elements
        for x in range(len(composition)):
            elements = composition[x]['elements'].keys()
            elements = [
                x if x not in data[reaction]['reaction']['element_substitution']
                else data[reaction]['reaction']['element_substitution'][x] for x in elements
            ]
            # print(elements)
            all_elements = all_elements + elements

    elem_set = set(all_elements)
    values = range(len(elem_set))
    elem_dict = dict(zip(elem_set, values))   # dictionary mapping element to value

    return elem_dict


def preprocess_target_stoich(data, elem_dict):
    """takes input data and element dictionary
    Returns: target in element encoding in numpy array
    """
    targets = []
    dodgy_doi = []
    dodgy_indices = []

    for reaction in range(len(data)):
        composition = data[reaction]['target']['composition']
        # iterate through each material in target and get list of dictionaries elem:stoich for each of them
        elements = [
            {
                element_: '('+stoich_+')*{}'.format(composition[x]["amount"])
                for element_, stoich_ in composition[x]['elements'].items()
            }
            for x in range(len(composition))
        ]

        # add dictionaries together into elements_full
        elements_full = {}
        for x in elements:
            for element, stoich in x.items():
                # check if element is substituted and replace if so
                if element in data[reaction]['reaction']['element_substitution']:
                    element = data[reaction]['reaction']['element_substitution'][element]

                if element in elements_full:
                    elements_full[element] = elements_full[element] + '+' + stoich
                else:
                    elements_full[element] = stoich

        # map elem dict to integers (dictionary comprehension)
        int_encoded = {elem_dict[elem]: val for elem, val in elements_full.items()}
        # map integer encoding to one hot encoding
        ohe_encoded = [0 for _ in range(len(elem_dict))]
        for element, stoich in int_encoded.items():
            # replace variable stoichs
            amount_vars = data[reaction]['target']['amounts_vars']
            for var, amount in amount_vars.items():

                # this replaces with average value
                if (amount['max_value'] is not None) and (amount['min_value'] is not None):
                    # assert (amount['max_value'] + amount['min_value']) / 2 is float
                    var_stoich = (amount['max_value'] + amount['min_value']) / 2
                else:
                    dodgy_doi.append(data[reaction]['doi'])
                    dodgy_indices.append(reaction)

                # put in missing division
                index = stoich.find(var)
                if index < len(stoich)-1:
                    if stoich[index+1].isdigit():
                        # print(stoich)
                        stoich = stoich[:index+1] + '/' + stoich[index+1:]
                        # print(stoich)
                stoich = stoich.replace(var, '('+str(var_stoich)+')')
            # if x not replaced
            if any(['x' in stoich, 'y' in stoich, 'z' in stoich, 'a' in stoich]):
                # print('before', stoich)
                # stoich = stoich.replace('x', '0.0')
                # stoich = stoich.replace('y', '0.0')
                # stoich = stoich.replace('z', '0.0')
                # stoich = stoich.replace('a', '0.0')
                dodgy_doi.append(data[reaction]['doi'])
                dodgy_indices.append(reaction)

            # evaluate
            try:
                ohe_encoded[element] = eval(stoich)
            except NameError:
                print(data[reaction]['targets_string'])
                print(stoich)
                dodgy_doi.append(data[reaction]['doi'])
                dodgy_indices.append(reaction)
            except ZeroDivisionError:
                print(data[reaction]['targets_string'])
                print(stoich)
                dodgy_doi.append(data[reaction]['doi'])
                dodgy_indices.append(reaction)
            if ohe_encoded[element] < 0:
                # print(data[reaction])
                print(stoich)
                dodgy_doi.append(data[reaction]['doi'])
                dodgy_indices.append(reaction)
        if np.count_nonzero(ohe_encoded) < 2:
            print(data[reaction])
            dodgy_doi.append(data[reaction]['doi'])

        # normalise
        ohe_encoded = normalise(ohe_encoded)
        targets.append(ohe_encoded)

    if dodgy_doi:
        print(f"BAD DOI: {set(dodgy_doi)}")

    if dodgy_indices:
        print(f"BAD IDX: {set(dodgy_indices)}")

    return np.array(targets)


def preprocess_precursors_stoich(data, elem_dict):
    """takes input data and element dictionary
    Returns: precursors in element encoding in numpy array
    Amounts of each precursor in numpy array
    """

    # initialise numpy array
    precursors = np.zeros((len(data), args.max_prec, len(elem_dict)))
    precursors_amounts = np.zeros((len(data), args.max_prec))
    dodgy_doi = []

    for reaction in range(len(data)):
        precursors_reaction = np.zeros((args.max_prec, len(elem_dict)))
        precursors_reaction_amounts = np.zeros(args.max_prec)

        for precursor in range(len(data[reaction]['precursors'])):
            composition = data[reaction]['precursors'][precursor]['composition']
            prec_formula = data[reaction]['precursors'][precursor]['material_formula']
            left_side = data[reaction]['reaction']['left_side']
            prec_amount = '1.0'
            for d in left_side:
                if d['material'] == prec_formula:
                    prec_amount = d['amount']
            # print(prec_amount)
            amount_vars = data[reaction]['target']['amounts_vars']
            for var, amount in amount_vars.items():

                # this replaces with average value
                if (amount['max_value'] is not None) and (amount['min_value'] is not None):
                    #assert (amount['max_value'] + amount['min_value']) / 2 is float
                    var_stoich = (amount['max_value'] + amount['min_value']) / 2
                else:
                    var_stoich = 0.0
                    dodgy_doi.append(data[reaction]['doi'])

                prec_amount = prec_amount.replace(var, str(var_stoich))
            # if x not replaced
            if 'x' in prec_amount:
                prec_amount = prec_amount.replace('x', '0.1')
                dodgy_doi.append(data[reaction]['doi'])

            prec_amount = eval(prec_amount)

            if prec_amount == 0:
                prec_amount = 1
                dodgy_doi.append(data[reaction]['doi'])
            elif prec_amount < 0:
                print(prec_amount, data[reaction]['doi'])
                dodgy_doi.append(data[reaction]['doi'])
                prec_amount = abs(prec_amount)

            # get list of dictionaries for each material in precursor
            elements = [{element_: '('+stoich_+')*{}'.format(composition[x]["amount"]) for element_, stoich_ in composition[x]['elements'].items()} for x in range(len(composition))]

            # add dictionaries together into elements_full
            elements_full = {}
            for x in range(len(elements)):
                for element, stoich in elements[x].items():

                    # check if element is substituted and replace if so
                    if element in data[reaction]['reaction']['element_substitution']:
                        element = data[reaction]['reaction']['element_substitution'][element]

                    # if it could be substituted for multiple elements, just take the first
                    if element in data[reaction]['precursors'][precursor]['elements_vars']:
                        element = data[reaction]['precursors'][precursor]['elements_vars'][element][0]

                    if element in elements_full:
                        elements_full[element] = elements_full[element] + '+' + stoich
                    else:
                        elements_full[element] = stoich

            # print(elements_full)
            # map dict to integers (dictionary comprehension)
            int_encoded = {elem_dict[elem]: val for elem, val in elements_full.items()}
            # print(int_encoded)
            # map integer encoding to one hot encoding
            ohe_encoded = [0 for _ in range(len(elem_dict))]
            for element, stoich in int_encoded.items():

                amount_vars = data[reaction]['precursors'][precursor]['amounts_vars']
                for var, amount in amount_vars.items():

                    # this replaces with average value
                    if (amount['max_value'] is not None) and (amount['min_value'] is not None):
                        #assert (amount['max_value'] + amount['min_value']) / 2 is float
                        var_stoich = (amount['max_value'] + amount['min_value']) / 2
                    else:
                        var_stoich = 0.0
                        dodgy_doi.append(data[reaction]['doi'])

                    index = stoich.find(var)
                    if index < len(stoich)-1:
                        if stoich[index+1].isdigit():
                            #print(stoich)
                            stoich = stoich[:index+1] + '/' + stoich[index+1:]
                            #print(stoich)
                    stoich = stoich.replace(var, '('+str(var_stoich)+')')
                    # if x not replaced
                if 'x' in stoich:
                    stoich = stoich.replace('x', '0.0')
                    dodgy_doi.append(data[reaction]['doi'])

                try:
                    ohe_encoded[element] = eval(stoich)
                except NameError:
                    print(stoich)
                    dodgy_doi.append(data[reaction]['doi'])
                if ohe_encoded[element] < 0:
                    print(stoich)
                    dodgy_doi.append(data[reaction]['doi'])
            # normalise
            ohe_encoded = normalise(ohe_encoded)
            precursors_reaction[precursor] = ohe_encoded
            precursors_reaction_amounts[precursor] = prec_amount
        precursors[reaction] = precursors_reaction
        precursors_amounts[reaction] = precursors_reaction_amounts

    if dodgy_doi: print(set(dodgy_doi))

    return precursors, precursors_amounts


def preprocess_precursors_magpie(data, elem_dict):
    """input data and elem_dict
    encodes for stoich first
    output magpie features for precursors
    """
    # stoich encoding
    prec_stoich, _ = preprocess_precursors_stoich(data, elem_dict)

    # convert stoich encoding to a clean formula
    vec_to_elem_dict = {v: k for k, v in elem_dict.items()}
    reaction_index = []
    prec_string = []
    for reaction in range(len(prec_stoich)):
        for precursor in prec_stoich[reaction]:
            decoded_prec = decode_material(precursor, vec_to_elem_dict)
            composition = dict_to_formula(decoded_prec)
            if '-' in composition:
                print(data[reaction])
                print(composition)
            reaction_index.append(reaction)
            prec_string.append(composition)


    d = {'reaction': reaction_index, 'composition': prec_string}
    df = pd.DataFrame(data=d)

    data = StrToComposition(target_col_id="composition_obj").featurize_dataframe(df, "composition")

    # Use the features from MAGPIE
    feature_calculators = MultipleFeaturizer([
        cf.Stoichiometry(),
        cf.ElementProperty.from_preset("magpie"),
        cf.ValenceOrbital(props=["avg"]),
        cf.IonProperty(fast=True)
    ])
    feature_labels = feature_calculators.feature_labels()

    data = feature_calculators.featurize_dataframe(data, col_id="composition_obj", ignore_errors=True)
    data = data.fillna(value=0)

    # group by reaction and make numpy array
    prec_magpie = np.zeros((len(prec_stoich), args.max_prec, len(feature_labels)))
    for reaction in range(len(prec_stoich)):
        reaction_df = data.loc[data['reaction'] == reaction]
        prec_magpie[reaction] = reaction_df[feature_labels].values

    return prec_magpie


def preprocess_precursors_roost(data, elem_dict, get_amount, get_all):
    """input: data and elem_dict
    encodes for stoich first
    output: magpie embedding dict for precursors
    list of list of tuples for precursors with amounts (formula, amount)
    If get_all flag is True, then returns all embeddings
    """
    # stoich encoding
    prec_stoich, prec_amounts = preprocess_precursors_stoich(data, elem_dict)

    # convert stoich encoding to a clean formula
    vec_to_elem_dict = {v: k for k, v in elem_dict.items()}
    reaction_index = []
    prec_string = []
    for reaction in range(len(prec_stoich)):
        for precursor in prec_stoich[reaction]:
            decoded_prec = decode_material(precursor, vec_to_elem_dict)
            composition = dict_to_formula(decoded_prec)
            if '-' in composition:
                print(data[reaction])
                print(composition)
            reaction_index.append(reaction)
            prec_string.append(composition)

    d = {'reaction': reaction_index, 'composition': prec_string}
    df = pd.DataFrame(data=d)
    data = StrToComposition(target_col_id="composition_obj").featurize_dataframe(df, "composition")

    # Use the features from MAGPIE
    feature_calculators = MultipleFeaturizer(
        [cf.Stoichiometry(),
        cf.ElementProperty.from_preset("magpie"),
        cf.ValenceOrbital(props=["avg"]),
        cf.IonProperty(fast=True)]
    )
    feature_labels = feature_calculators.feature_labels()

    data = feature_calculators.featurize_dataframe(data, col_id="composition_obj", ignore_errors=True)
    data = data.fillna(value=0)

    # Get precursor magpie dictionary. T transposes dataframe
    prec_magpie = data[['composition', *feature_labels]]
    prec_magpie_dict = prec_magpie.set_index('composition').T.to_dict('list')

    # group by reaction
    formulas = data.groupby('reaction')['composition'].apply(list)
    print(formulas)

    if get_all:
        prec_magpie_pre = np.zeros((len(prec_stoich), args.max_prec, len(feature_labels)))
        for reaction in range(len(prec_stoich)):
            reaction_df = data.loc[data['reaction'] == reaction]
            prec_magpie_pre[reaction] = reaction_df[feature_labels].values

        prec_roost = []
        prec_roost_am = []
        for reaction, precs in formulas.iteritems():
            prec_roost.append([i for i in precs if i])
            prec_roost_am.append([(precs[i], prec_amounts[reaction][i]) for i in range(len(precs)) if precs[i]])

        return prec_stoich, prec_magpie_pre, prec_roost, prec_roost_am, prec_magpie_dict
    else:
        # remove empty strings
        prec_formulas = []
        for reaction, precs in formulas.iteritems():
            if get_amount:
                prec_formulas.append([(precs[i], prec_amounts[reaction][i]) for i in range(len(precs)) if precs[i]])
            else:
                prec_formulas.append([i for i in precs if i])

        return prec_formulas, prec_magpie_dict


def augment_data(sources, targets):
    """Augment dataset with rearrangments of precursors.
    """
    augmented_sources = []
    augmented_targets = []

    for i in range(len(sources)):
        perms = permutations(sources[i])
        for perm in perms:
            augmented_sources.append(perm)
            augmented_targets.append(targets[i])

    return np.array(augmented_sources), np.array(augmented_targets)


def remove_rare_elems(data, precs, targets, elem_dict):
    """Find elements to remove from dataset. Keep only top args.num_elem
    Returns: data_trimmed (list of dicts)
    """
    vec_to_elem_dict = {v: k for k, v in elem_dict.items()}
    total_elems = np.sum(precs, axis=1) + np.array(targets)
    elem_dist = np.count_nonzero(total_elems, axis=0)
    elem_dist = [(vec_to_elem_dict[i], elem_dist[i]) for i in range(len(elem_dist))]
    elem_dist_sorted = sorted(elem_dist, key=lambda x: x[1], reverse=True)
    print(elem_dist_sorted)
    rare_elems = [x[0] for x in elem_dist_sorted[args.num_elem:]]

    data_trimmed = []
    for reaction in range(len(total_elems)):
        is_rare_elem = [(vec_to_elem_dict[x] in rare_elems) for x in np.nonzero(total_elems[reaction])[0]]
        if any(is_rare_elem):
            pass
        else:
            data_trimmed.append(data[reaction])
    print(len(data_trimmed))

    return data_trimmed


def preprocess_actions(data):
    """Returns:
    - action_dict: a dict mapping actions to integers (corresponding to OHE vector columns)
    - ohe: OHE lists for each action sequence
    """
    action_sequences = []

    for reaction in range(len(data)):

        operations = data[reaction]['operations']
        actions = [operations[x]['type'] for x in range(len(operations))]       # action type
        # actions = [operations[x]['string'] for x in range(len(operations))]    # action string
        action_sequences.append(actions)

    # get dict of action sequences - mapping a process to an int.
    action_sequence_set = set(item for t in action_sequences for item in t)
    values = range(len(action_sequence_set))
    action_dict = dict(zip(action_sequence_set, values))
    print(action_dict)

    # integer encode actions
    action_sequences_encoded = []
    for reaction in range(len(action_sequences)):
        action_sequences_encoded.append([action_dict[item] for item in action_sequences[reaction]])

    # OHE actions
    ohe = []
    for reaction in range(len(action_sequences_encoded)):
        ohe_reaction = []
        # loop over each action in sequence
        for action in range(len(action_sequences_encoded[reaction])):
            ohe_action = [0 for _ in range(len(action_dict))]
            # set bit corresponding to int to 1
            ohe_action[action_sequences_encoded[reaction][action]] = 1
            ohe_reaction.append(ohe_action)
        ohe.append(ohe_reaction)

    return ohe, action_dict


def save_dataset(data, path):
    """Writes preprocessed data to pickle files
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(f'{path}{args.ps}.pkl', 'wb') as f:
        pkl.dump(data, f)

    print(f'Dumped to {path}{args.ps}.pkl')


def save_dataset_splits(data, path):
    """Writes preprocessed data to pickle files
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    for num in data["n_prec"].unique():
        with open(f'{path}_np_{num}{args.ps}.pkl', 'wb') as f:
            pkl.dump(data[data["n_prec"] == num], f)


def save_dict(data, path):
    """Writes dict to json files
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(f'{path}{args.ps}.json', 'w') as f:
        json.dump(data, f)

    print(f'Dumped to {path}{args.ps}.json')


def build_and_save_df():
    """Assembles all data together into a pandas DataFrame
    and saves as csv.
    Saves
    ----------------
    elem_dict:
        dict
        Dictionary of mapping stoich vector to elements
    magpie_embed:
        dict
        Dictionary mapping precursors to magpie embeddings (for use in Roost)
    df:
        pd.Dataframe
        Dataframe with each row giving reaction information

        Data columns are as follows:
        ==========      ==============================================================
        dois            (string) reaction doi (for reference)
        reaction        (string) reaction string
        prec_stoich     (list of lists) stoichiometry vector for each precursor (zero padded)
        prec_magpie     (list of lists) magpie embedding vector for each precursor
        prec_roost      (list of strings) normalised formulas for each precursor for Roost
        prec_roost_am   (list of tuples) as above, except (formula, amount) for each instead
        actions         (list of lists) OHE action sequences for the reaction
        target          (list) stoichiometry vector for target material
        ==========      ==============================================================
    """

    # NOTE the fact the the numbers for each element change each time the file
    # is processed should be fixed.

    data = load_dataset()

    elem_dict = find_elem_dict(data)
    save_dict(elem_dict, args.elem_dict)

    # data = data[:500]

    targets_stoich = preprocess_target_stoich(data, elem_dict)
    # prec_stoich, _ = preprocess_precursors_stoich(data, elem_dict)

    # # reduce dataset
    # if args.num_elem > 0:
    #     data = remove_rare_elems(data, prec_stoich, targets_stoich, elem_dict)
    #     elem_dict = find_elem_dict(data)
    #     print(elem_dict)
    #     targets_stoich = preprocess_target_stoich(data, elem_dict)

    # NOTE the prec_roost column is unneccesary and should be changed
    # NOTE the prec_magpie column is unneccesary with the associated magpie_embed reference and should be changed

    processed = preprocess_precursors_roost(data, elem_dict, get_amount=False, get_all=True)
    prec_stoich, prec_magpie, prec_roost, prec_roost_am, magpie_embed = processed

    save_dict(magpie_embed, args.magpie_embed)

    actions, action_dict = preprocess_actions(data)
    save_dict(action_dict, args.action_dict)

    dois = [x['doi'] for x in data]
    reactions = [x['reaction_string'] for x in data]

    features = {'prec_stoich': prec_stoich,
                'prec_magpie': prec_magpie,
                'prec_roost': prec_roost,
                'prec_roost_am': prec_roost_am,
                'actions': actions,
                'target': targets_stoich}

    features = {k: v.tolist() if type(v) == np.ndarray else v for (k, v) in features.items()}

    # save data
    df = pd.DataFrame({'dois': dois, 'reaction': reactions, **features})
    df["n_prec"] = df["prec_roost"].apply(lambda x: len(x))

    save_dataset(df, args.clean_set)

    df_train, df_test = split(df, random_state=args.seed, test_size=args.test_size)
    save_dataset(df_test, args.test_set)

    if args.val_size:
        df_train, df_val = split(df_train, random_state=args.seed, test_size=args.val_size / (1 - args.test_size))
        save_dataset(df_val, args.val_set)

    save_dataset(df_train, args.train_set)

    if args.split_prec_amts:
        save_dataset_splits(df_train, args.train_set)
        save_dataset_splits(df_test, args.test_set)

        if args.val_size:
            save_dataset_splits(df_val, args.val_set)


if __name__ == "__main__":

    args = parse_args()

    build_and_save_df()
