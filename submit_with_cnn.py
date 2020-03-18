"""
This module is a part of system for the automatic enrichment
of a WordNet-like taxonomy.

Copyright 2020 Ivan Bondarenko

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from argparse import ArgumentParser
import codecs
import os
import pickle
import random
from typing import Set

import nltk
import numpy as np
import tensorflow as tf

import ruwordnet_parsing
import trainset_preparing
import hyponyms_loading
import neural_network


def load_existing_predictions(file_name: str) -> Set[tuple]:
    line_idx = 1
    hyponyms = set()
    with codecs.open(file_name, mode="r", encoding="utf-8", errors="ignore") as fp:
        cur_line = fp.readline()
        while len(cur_line) > 0:
            prep_line = cur_line.strip()
            if len(prep_line) > 0:
                err_msg = 'File `{0}`: line {1} is wrong!'.format(file_name, line_idx)
                line_parts = list(filter(lambda it2: len(it2) > 0, map(lambda it1: it1.strip(), prep_line.split('\t'))))
                assert len(line_parts) == 3, err_msg
                tokens = tuple(filter(
                    lambda it2: len(it2) > 0,
                    map(lambda it1: it1.strip(), line_parts[0].lower().split())
                ))
                assert len(tokens) > 0, err_msg
                hyponyms.add(tokens)
            cur_line = fp.readline()
            line_idx += 1
    return hyponyms


def does_neural_network_exist(file_name: str) -> bool:
    dir_name = os.path.dirname(os.path.normpath(file_name))
    if len(dir_name) == 0:
        ok = True
    else:
        ok = os.path.isdir(dir_name)
    if ok:
        if (not os.path.isfile(file_name)) and (not os.path.isdir(file_name)):
            files_list = list(filter(
                lambda it: it.startswith(file_name),
                os.listdir('.' if len(dir_name) == 0 else dir_name)
            ))
            if len(files_list) == 0:
                ok = False
    return ok


def main():
    random.seed(142)
    np.random.seed(142)
    tf.random.set_seed(142)

    parser = ArgumentParser()
    parser.add_argument('-t', '--track', dest='track_name', type=str, required=True, choices=['nouns', 'verbs'],
                        help='A competition track name (nouns or verbs).')
    parser.add_argument('-f', '--fasttext', dest='fasttext_name', type=str, required=True,
                        help='A binary file with a Facebook-like FastText model (*.bin).')
    parser.add_argument('-w', '--wordnet', dest='wordnet_dir', type=str, required=True,
                        help='A directory with unarchived RuWordNet.')
    parser.add_argument('-i', '--input', dest='input_data_dir', type=str, required=True,
                        help='A directory with input data, i.e. lists of unseen hyponyms for public and private '
                             'submission.')
    parser.add_argument('-o', '--output', dest='output_data_dir', type=str, required=True,
                        help='A directory with output data, i.e. lists of unseen hyponyms and their hypernyms, found '
                             'as a result of this program execution, for public and private submission.')
    parser.add_argument('-c', '--cache_dir', dest='cache_dir', type=str, required=True,
                        help='A directory with cached data for training.')
    parser.add_argument('-n', '--number', dest='number_of_hypernyms', type=int, required=False, default=10,
                        help='A number of hypernyms generated for each unseen hyponym.')
    parser.add_argument('--conv', dest='conv_size', type=int, required=False, default=512,
                        help='A number of feature maps in a 1D convolution layer for each convolution window.')
    parser.add_argument('--n_hidden', dest='hidden_layers_number', type=int, required=False, default=1,
                        help='A number of hidden dense layers after the convolution layer.')
    parser.add_argument('--hidden_size', dest='hidden_layer_size', type=int, required=False, default=4096,
                        help='Size of each hidden dense layer.')
    parser.add_argument('--dropout', dest='dropout_rate', type=float, required=False, default=0.5,
                        help='A fraction of the input units to drop for the dropout technique.')
    parser.add_argument('--lr_max', dest='max_learning_rate', type=float, required=False, default=1e-3,
                        help='A maximal learning rate for the cyclical learning rate schedule.')
    parser.add_argument('--lr_min', dest='min_learning_rate', type=float, required=False, default=1e-5,
                        help='A minimal learning rate for the cyclical learning rate schedule.')
    parser.add_argument('--cycle_length', dest='training_cycle_length', type=int, required=False, default=11,
                        help='A period or cycle length for the cyclical learning rate schedule.')
    parser.add_argument('--epochs', dest='max_epochs', type=int, required=False, default=1000,
                        help='A maximal number of epochs to train the neural network.')
    parser.add_argument('--batch', dest='batch_size', type=int, required=False, default=128, help='A mini-batch size.')
    parser.add_argument('--bayesian', dest='bayesian_nn', action='store_true',
                        help='Must a Bayesian neural network be used?')
    parser.add_argument('--monte_carlo', dest='num_monte_carlo', type=int, required=False, default=20,
                        help='A sample number for the Monte Carlo inference in a bayesian neural network.')
    parser.add_argument('--kl_weight', dest='kl_weight', type=float, required=False, default=0.5,
                        help='Weight of the KL loss for Bayesian deep learning.')
    args = parser.parse_args()

    assert args.number_of_hypernyms >= 10, \
        '{0} is too small value for the hypernyms number!'.format(args.number_of_hypernyms)
    is_bayesian = args.bayesian_nn
    if is_bayesian:
        num_monte_carlo = args.num_monte_carlo
        assert num_monte_carlo > 1
    else:
        num_monte_carlo = 0
    cached_data_dir = os.path.normpath(args.cache_dir)
    assert os.path.isdir(cached_data_dir)
    nltk.download('punkt')
    wordnet_dir = os.path.normpath(args.wordnet_dir)
    assert os.path.isdir(wordnet_dir)
    wordnet_senses_name = os.path.join(wordnet_dir, 'senses.N.xml' if args.track_name == 'nouns' else 'senses.V.xml')
    wordnet_synsets_name = os.path.join(wordnet_dir, 'synsets.N.xml' if args.track_name == 'nouns' else 'synsets.V.xml')
    wordnet_relations_name = os.path.join(
        wordnet_dir,
        'synset_relations.N.xml' if args.track_name == 'nouns' else 'synset_relations.V.xml'
    )
    assert os.path.isfile(wordnet_senses_name)
    assert os.path.isfile(wordnet_synsets_name)
    assert os.path.isfile(wordnet_relations_name)
    fasttext_model_path = os.path.normpath(args.fasttext_name)
    assert os.path.isfile(fasttext_model_path)
    input_data_dir = os.path.normpath(args.input_data_dir)
    os.path.isdir(input_data_dir), 'Directory `{0}` does not exist!'.format(input_data_dir)
    output_data_dir = os.path.normpath(args.output_data_dir)
    os.path.isdir(output_data_dir), 'Directory `{0}` does not exist!'.format(output_data_dir)
    public_data_name = os.path.join(input_data_dir,
                                    '{0}_public.tsv'.format('nouns' if args.track_name == 'nouns' else 'verbs'))
    assert os.path.isfile(public_data_name), 'File `{0}` does not exist!'.format(public_data_name)
    public_submission_name = os.path.join(output_data_dir,
                                          'submitted_{0}_public.tsv'.format(('nouns' if args.track_name == 'nouns'
                                                                             else 'verbs')))
    private_data_name = os.path.join(input_data_dir,
                                     '{0}_private.tsv'.format('nouns' if args.track_name == 'nouns' else 'verbs'))
    assert os.path.isfile(private_data_name), 'File `{0}` does not exist!'.format(private_data_name)
    private_submission_name = os.path.join(output_data_dir,
                                           'submitted_{0}_private.tsv'.format(('nouns' if args.track_name == 'nouns'
                                                                               else 'verbs')))

    synsets = ruwordnet_parsing.load_synsets(senses_file_name=wordnet_senses_name,
                                             synsets_file_name=wordnet_synsets_name)
    data_for_public_submission = hyponyms_loading.load_terms_for_submission(public_data_name)
    print('Number of hyponyms for public submission is {0}.'.format(len(data_for_public_submission)))
    data_for_private_submission = hyponyms_loading.load_terms_for_submission(private_data_name)
    print('Number of hyponyms for private submission is {0}.'.format(len(data_for_private_submission)))
    print('')

    solver_name = os.path.join(cached_data_dir, 'fasttext_and_{0}_cnn.h5'.format(
        'bayesian' if is_bayesian else 'simple'))
    solver_params_name = os.path.join(cached_data_dir, 'fasttext_and_{0}_cnn_params.pkl'.format(
        'bayesian' if is_bayesian else 'simple'))
    if does_neural_network_exist(solver_name) and os.path.isfile(solver_params_name):
        with open(solver_params_name, 'rb') as fp:
            (max_hyponym_length, max_hypernym_length, all_tokens, embeddings_matrix, conv_size, hidden_layer_size,
             hidden_layers_number) = pickle.load(fp)
        assert isinstance(embeddings_matrix, np.ndarray)
        assert len(embeddings_matrix.shape) == 2
        assert embeddings_matrix.shape[0] == (len(all_tokens) + 1)
        if is_bayesian:
            solver = neural_network.build_bayesian_cnn(
                max_hyponym_length=max_hyponym_length, max_hypernym_length=max_hypernym_length,
                word_embeddings=embeddings_matrix, n_feature_maps=args.conv_size,
                hidden_layer_size=args.hidden_layer_size, n_hidden_layers=args.hidden_layers_number,
                n_train_samples=100, kl_weight=args.kl_weight
            )
        else:
            solver = neural_network.build_cnn(
                max_hyponym_length=max_hyponym_length, max_hypernym_length=max_hypernym_length,
                word_embeddings=embeddings_matrix, n_feature_maps=args.conv_size,
                hidden_layer_size=args.hidden_layer_size,
                n_hidden_layers=args.hidden_layers_number, dropout_rate=args.dropout_rate
            )
        solver.load_weights(solver_name)
        print('The neural network has been loaded from the `{0}`...'.format(solver_name))
    else:
        data_for_training, data_for_validation, data_for_testing = ruwordnet_parsing.prepare_data_for_training(
            senses_file_name=wordnet_senses_name, synsets_file_name=wordnet_synsets_name,
            relations_file_name=wordnet_relations_name
        )
        all_tokens = ruwordnet_parsing.tokens_from_synsets(synsets, additional_sources=[data_for_public_submission,
                                                                                        data_for_private_submission])
        print('Vocabulary size is {0}.'.format(len(all_tokens)))
        embeddings_matrix = trainset_preparing.calculate_word_embeddings(all_tokens, fasttext_model_path)
        print('All word embeddings are calculated...')
        max_hyponym_length, max_hypernym_length = trainset_preparing.get_maximal_lengths_of_texts(synsets)
        print('Maximal length of a single hyponym is {0}.'.format(max_hyponym_length))
        print('Maximal length of a single hypernym is {0}.'.format(max_hypernym_length))
        print('')

        X_train, y_train = trainset_preparing.build_dataset_for_training(
            data=data_for_training, synsets=synsets, tokens_dict=all_tokens,
            max_hyponym_length=max_hyponym_length, max_hypernym_length=max_hypernym_length
        )
        X_val, y_val = trainset_preparing.build_dataset_for_training(
            data=data_for_validation, synsets=synsets, tokens_dict=all_tokens,
            max_hyponym_length=max_hyponym_length, max_hypernym_length=max_hypernym_length
        )
        if is_bayesian:
            solver = neural_network.build_bayesian_cnn(
                max_hyponym_length=max_hyponym_length, max_hypernym_length=max_hypernym_length,
                word_embeddings=embeddings_matrix, n_feature_maps=args.conv_size,
                hidden_layer_size=args.hidden_layer_size, n_hidden_layers=args.hidden_layers_number,
                n_train_samples=y_train.shape[0], kl_weight=args.kl_weight
            )
        else:
            solver = neural_network.build_cnn(
                max_hyponym_length=max_hyponym_length, max_hypernym_length=max_hypernym_length,
                word_embeddings=embeddings_matrix, n_feature_maps=args.conv_size,
                hidden_layer_size=args.hidden_layer_size,
                n_hidden_layers=args.hidden_layers_number, dropout_rate=args.dropout_rate
            )
        solver = neural_network.train_neural_network(
            X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
            neural_network=solver, max_epochs=args.max_epochs, training_cycle_length=args.training_cycle_length,
            max_learning_rate=args.max_learning_rate, min_learning_rate=args.min_learning_rate,
            is_bayesian=is_bayesian, num_monte_carlo=num_monte_carlo, batch_size=args.batch_size
        )
        del X_train, y_train, X_val, y_val
        X_test, y_test = trainset_preparing.build_dataset_for_training(
            data=data_for_testing, synsets=synsets, tokens_dict=all_tokens,
            max_hyponym_length=max_hyponym_length, max_hypernym_length=max_hypernym_length
        )
        neural_network.evaluate_neural_network(
            X=X_test, y_true=y_test,
            neural_network=solver, batch_size=args.batch_size,
            num_monte_carlo=num_monte_carlo
        )
        del X_test, y_test
        with open(solver_params_name, 'wb') as fp:
            pickle.dump((max_hyponym_length, max_hypernym_length, all_tokens, embeddings_matrix, args.conv_size,
                         args.hidden_layer_size, args.hidden_layers_number), fp, protocol=pickle.HIGHEST_PROTOCOL)
        solver.save_weights(solver_name, overwrite=True)

    n_data_parts = 20
    print('Public submission is started...')
    if os.path.isfile(public_submission_name):
        predicted_hyponyms = load_existing_predictions(public_submission_name)
    else:
        predicted_hyponyms = set()
    with codecs.open(public_submission_name, mode='a', encoding='utf-8', errors='ignore') as fp:
        data_part_size = int(np.ceil(len(data_for_public_submission) / n_data_parts))
        data_part_counter = 0
        for idx, hyponym in enumerate(data_for_public_submission):
            if hyponym not in predicted_hyponyms:
                dataset, synset_IDs = trainset_preparing.build_dataset_for_submission(
                    unseen_hyponym=hyponym, synsets=synsets, tokens_dict=all_tokens,
                    max_hyponym_length=max_hyponym_length, max_hypernym_length=max_hypernym_length
                )
                probabilities = neural_network.apply_neural_network(
                    X=dataset, neural_network=solver,
                    batch_size=args.batch_size, num_monte_carlo=num_monte_carlo
                )
                del dataset
                best_synsets = list(map(lambda idx: (synset_IDs[idx], probabilities[idx]), range(len(synset_IDs))))
                best_synsets.sort(key=lambda it: (-it[1], it[0]))
                if len(best_synsets) > args.number_of_hypernyms:
                    n = args.number_of_hypernyms
                else:
                    n = len(best_synsets)
                for synset_id in map(lambda it: it[0], best_synsets[0:n]):
                    fp.write('{0}\t{1}\t{2}\n'.format(' '.join(hyponym).upper(), synset_id, synsets[synset_id][2]))
                del synset_IDs, best_synsets
            if (idx + 1) % data_part_size == 0:
                data_part_counter += 1
                print('  {0} % of public data have been processed...'.format(data_part_counter * 5))
        if data_part_counter < n_data_parts:
            print('  100 % of public data have been processed...')
    print('Public submission is finished...')
    print('')
    print('Private submission is started...')
    if os.path.isfile(private_submission_name):
        predicted_hyponyms = load_existing_predictions(private_submission_name)
    else:
        predicted_hyponyms = set()
    with codecs.open(private_submission_name, mode='a', encoding='utf-8', errors='ignore') as fp:
        data_part_size = int(np.ceil(len(data_for_private_submission) / n_data_parts))
        data_part_counter = 0
        for idx, hyponym in enumerate(data_for_private_submission):
            if hyponym not in predicted_hyponyms:
                dataset, synset_IDs = trainset_preparing.build_dataset_for_submission(
                    unseen_hyponym=hyponym, synsets=synsets, tokens_dict=all_tokens,
                    max_hyponym_length=max_hyponym_length, max_hypernym_length=max_hypernym_length
                )
                probabilities = neural_network.apply_neural_network(
                    X=dataset, neural_network=solver,
                    batch_size=args.batch_size, num_monte_carlo=num_monte_carlo
                )
                del dataset
                best_synsets = list(map(lambda idx: (synset_IDs[idx], probabilities[idx]), range(len(synset_IDs))))
                best_synsets.sort(key=lambda it: (-it[1], it[0]))
                if len(best_synsets) > args.number_of_hypernyms:
                    n = args.number_of_hypernyms
                else:
                    n = len(best_synsets)
                for synset_id in map(lambda it: it[0], best_synsets[0:n]):
                    fp.write('{0}\t{1}\t{2}\n'.format(' '.join(hyponym).upper(), synset_id, synsets[synset_id][2]))
                del synset_IDs, best_synsets
            if (idx + 1) % data_part_size == 0:
                data_part_counter += 1
                print('  {0} % of private data have been processed...'.format(data_part_counter * 5))
        if data_part_counter < n_data_parts:
            print('  100 % of private data have been processed...')
    print('Private submission is finished...')
    print('')


if __name__ == '__main__':
    main()
