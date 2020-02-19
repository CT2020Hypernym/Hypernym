from argparse import ArgumentParser
import codecs
import csv
import os
import pickle
import random

import nltk
import numpy as np
import tensorflow as tf

import ruwordnet_parsing
import trainset_preparing
import hyponyms_loading
import neural_network


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
    parser.add_argument('-b', '--public', dest='public_data', type=str, required=True,
                        help='A text file with a list of unseen hyponyms for public submission.')
    parser.add_argument('-r', '--private', dest='private_data', type=str, required=True,
                        help='A text file with a list of unseen hyponyms for private submission.')
    parser.add_argument('-c', '--cache_dir', dest='cache_dir', type=str, required=False, default=None,
                        help='A directory with cached data for training.')
    parser.add_argument('--conv', dest='conv_size', type=int, required=False, default=512,
                        help='A number of feature maps in a 1D convolution layer for each convolution window.')
    parser.add_argument('--n_hidden', dest='hidden_layers_number', type=int, required=False, default=1,
                        help='A number of hidden dense layers after the convolution layer.')
    parser.add_argument('--hidden_size', dest='hidden_layer_size', type=int, required=False, default=2048,
                        help='Size of each hidden dense layer.')
    parser.add_argument('--dropout', dest='dropout_rate', type=float, required=False, default=0.5,
                        help='A fraction of the input units to drop for the dropout technique.')
    parser.add_argument('--lr_max', dest='max_learning_rate', type=float, required=False, default=1e-3,
                        help='A maximal learning rate for the cyclical learning rate schedule.')
    parser.add_argument('--lr_min', dest='min_learning_rate', type=float, required=False, default=1e-6,
                        help='A minimal learning rate for the cyclical learning rate schedule.')
    parser.add_argument('--cycle_length', dest='training_cycle_length', type=int, required=False, default=9,
                        help='A period or cycle length for the cyclical learning rate schedule.')
    parser.add_argument('--epochs', dest='max_epochs', type=int, required=False, default=1000,
                        help='A maximal number of epochs to train the neural network.')
    parser.add_argument('--batch', dest='batch_size', type=int, required=False, default=512, help='A mini-batch size.')
    parser.add_argument('--bayesian', dest='bayesian_nn', action='store_true',
                        help='Must a Bayesian neural network be used?')
    parser.add_argument('--monte_carlo', dest='num_monte_carlo', type=int, required=False, default=10,
                        help='A sample number for the Monte Carlo inference in a bayesian neural network.')
    parser.add_argument('--kl_weight', dest='kl_weight', type=float, required=False, default=1e-1,
                        help='Weight of the KL loss for Bayesian deep learning.')
    args = parser.parse_args()

    is_bayesian = args.bayesian_nn
    if is_bayesian:
        num_monte_carlo = args.num_monte_carlo
        assert num_monte_carlo > 1
    else:
        num_monte_carlo = 0
    cached_data_dir = None if args.cache_dir is None else os.path.normpath(args.cache_dir)
    if cached_data_dir is not None:
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
    public_data_name = os.path.normpath(args.public_data)
    assert os.path.isfile(public_data_name)
    public_submission_name = os.path.join(os.path.dirname(public_data_name),
                                          'submitted_' + os.path.basename(public_data_name))
    private_data_name = os.path.normpath(args.private_data)
    assert os.path.isfile(private_data_name)
    private_submission_name = os.path.join(os.path.dirname(private_data_name),
                                           'submitted_' + os.path.basename(private_data_name))

    synsets = ruwordnet_parsing.load_synsets(senses_file_name=wordnet_senses_name,
                                             synsets_file_name=wordnet_synsets_name)
    data_for_training, data_for_validation, data_for_testing = ruwordnet_parsing.prepare_data_for_training(
        senses_file_name=wordnet_senses_name, synsets_file_name=wordnet_synsets_name,
        relations_file_name=wordnet_relations_name
    )
    data_for_public_submission = hyponyms_loading.load_terms_for_submission(public_data_name)
    print('Number of hyponyms for public submission is {0}.'.format(len(data_for_public_submission)))
    data_for_private_submission = hyponyms_loading.load_terms_for_submission(private_data_name)
    print('Number of hyponyms for private submission is {0}.'.format(len(data_for_private_submission)))
    print('')

    all_tokens = ruwordnet_parsing.tokens_from_synsets(synsets, additional_sources=[data_for_public_submission,
                                                                                    data_for_private_submission])
    print('Vocabulary size is {0}.'.format(len(all_tokens)))
    if cached_data_dir is None:
        embeddings_matrix = trainset_preparing.calculate_word_embeddings(all_tokens, fasttext_model_path)
    else:
        embeddings_matrix_filename = os.path.join(cached_data_dir, 'fasttext_embeddings.pkl')
        if os.path.isfile(embeddings_matrix_filename):
            with open(embeddings_matrix_filename, 'rb') as fp:
                embeddings_matrix = pickle.load(fp)
            assert isinstance(embeddings_matrix, np.ndarray)
            assert len(embeddings_matrix.shape) == 2
            assert embeddings_matrix.shape[0] == (len(all_tokens) + 1)
        else:
            embeddings_matrix = trainset_preparing.calculate_word_embeddings(all_tokens, fasttext_model_path)
            with open(embeddings_matrix_filename, 'wb') as fp:
                pickle.dump(embeddings_matrix, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print('All word embeddings are calculated...')
    max_hyponym_length, max_hypernym_length = trainset_preparing.get_maximal_lengths_of_texts(synsets)
    print('Maximal length of a single hyponym is {0}.'.format(max_hyponym_length))
    print('Maximal length of a single hypernym is {0}.'.format(max_hypernym_length))
    print('')

    trainset_generator = trainset_preparing.TrainsetGenerator(
        data_for_training=data_for_training, synsets=synsets, tokens_dict=all_tokens,
        max_hyponym_length=max_hyponym_length, max_hypernym_length=max_hypernym_length,
        batch_size=args.batch_size, deterministic=False
    )
    validset_generator = trainset_preparing.TrainsetGenerator(
        data_for_training=data_for_validation, synsets=synsets, tokens_dict=all_tokens,
        max_hyponym_length=max_hyponym_length, max_hypernym_length=max_hypernym_length,
        batch_size=args.batch_size, deterministic=True
    )
    if is_bayesian:
        solver = neural_network.build_bayesian_cnn(
            max_hyponym_length=max_hyponym_length, max_hypernym_length=max_hypernym_length,
            word_embeddings=embeddings_matrix, n_feature_maps=args.conv_size, hidden_layer_size=args.hidden_layer_size,
            n_hidden_layers=args.hidden_layers_number, n_train_samples=len(trainset_generator) * args.batch_size,
            kl_weight=args.kl_weight
        )
    else:
        solver = neural_network.build_cnn(
            max_hyponym_length=max_hyponym_length, max_hypernym_length=max_hypernym_length,
            word_embeddings=embeddings_matrix, n_feature_maps=args.conv_size, hidden_layer_size=args.hidden_layer_size,
            n_hidden_layers=args.hidden_layers_number, dropout_rate=args.dropout_rate
        )
    solver = neural_network.train_neural_network(
        data_for_training=trainset_generator, data_for_validation=validset_generator,
        neural_network=solver, max_epochs=args.max_epochs, training_cycle_length=args.training_cycle_length,
        max_learning_rate=args.max_learning_rate, min_learning_rate=args.min_learning_rate,
        is_bayesian=is_bayesian, num_monte_carlo=num_monte_carlo
    )
    del trainset_generator, validset_generator
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

    n_data_parts = 20
    print('Public submission is started...')
    with codecs.open(public_submission_name, mode='w', encoding='utf-8', errors='ignore') as fp:
        data_writer = csv.writer(fp, delimiter='\t', quotechar='"')
        data_part_size = int(np.ceil(len(data_for_public_submission) / n_data_parts))
        data_part_counter = 0
        for idx, hyponym in enumerate(data_for_public_submission):
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
            if len(best_synsets) > 10:
                n = 10
            else:
                n = len(best_synsets)
            for synset_id in map(lambda it: it[0], best_synsets[0:n]):
                data_writer.writerow([' '.join(hyponym).upper(), synset_id])
            if (idx + 1) % data_part_size == 0:
                data_part_counter += 1
                print('  {0} % of public data have been processed...'.format(data_part_counter * 5))
            del synset_IDs, best_synsets
        if data_part_counter < n_data_parts:
            print('  100 % of public data have been processed...')
    print('Public submission is finished...')
    print('')
    print('Private submission is started...')
    with codecs.open(private_submission_name, mode='w', encoding='utf-8', errors='ignore') as fp:
        data_writer = csv.writer(fp, delimiter='\t', quotechar='"')
        data_part_size = int(np.ceil(len(data_for_private_submission) / n_data_parts))
        data_part_counter = 0
        for idx, hyponym in enumerate(data_for_private_submission):
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
            if len(best_synsets) > 10:
                n = 10
            else:
                n = len(best_synsets)
            for synset_id in map(lambda it: it[0], best_synsets[0:n]):
                data_writer.writerow([' '.join(hyponym).upper(), synset_id])
            if (idx + 1) % data_part_size == 0:
                data_part_counter += 1
                print('  {0} % of private data have been processed...'.format(data_part_counter * 5))
            del synset_IDs, best_synsets
        if data_part_counter < n_data_parts:
            print('  100 % of private data have been processed...')
    print('Private submission is finished...')
    print('')


if __name__ == '__main__':
    main()
