from argparse import ArgumentParser
import codecs
import gc
import multiprocessing
import os
import pickle
import random
from typing import Dict, List, Tuple
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from bert.tokenization.bert_tokenization import FullTokenizer
import nltk
import numpy as np
from params_flow.optimizers import RAdam
import tensorflow as tf

import ruwordnet_parsing
import trainset_preparing
import hyponyms_loading
import bert_based_nn
import text_processing


def do_submission(submission_result_name: str,
                  input_hyponyms: List[Tuple[tuple, List[Tuple[str, str]]]],
                  occurrences_of_input_hyponyms: Dict[str, Dict[str, List[Tuple[str, Tuple[int, int]]]]],
                  synsets_from_wordnet: Dict[str, Tuple[List[str], str]], source_senses_from_wordnet: Dict[str, str],
                  inflected_senses_from_wordnet: Dict[str, Dict[str, Tuple[tuple, Tuple[int, int]]]],
                  bert_tokenizer: FullTokenizer, neural_network: tf.keras.Model, max_seq_len: int, batch_size: int,
                  num_monte_carlo: int = 0):
    if num_monte_carlo > 0:
        print('A sample number for the Monte Carlo inference is {0}.'.format(num_monte_carlo))
    hyponyms_with_hypernym_candidates = dict()
    for hyponym_value, hypernym_candidates in input_hyponyms:
        err_msg = 'The hyponym `{0}` is duplicated!'.format(' '.join(hyponym_value))
        assert hyponym_value in hyponyms_with_hypernym_candidates, err_msg
        candidate_IDs = set()
        for synset_id, hypernym_text in hypernym_candidates:
            assert synset_id in synsets_from_wordnet, 'Synset ID `{0}` is unknown!'.format(synset_id)
            assert synset_id in candidate_IDs, 'Synset ID `{0}` is duplicated!'.format(synset_id)
            candidate_IDs.add(synset_id)
        hyponyms_with_hypernym_candidates[hyponym_value] = sorted(list(candidate_IDs))
    n_processes = os.cpu_count()
    if n_processes > 1:
        pool = multiprocessing.Pool(processes=n_processes)
    else:
        pool = None
    with codecs.open(submission_result_name, mode='w', encoding='utf-8', errors='ignore') as fp:
        for hyponym_idx, (hyponym_value, _) in enumerate(input_hyponyms):
            print('Unseen hyponym `{0}`:'.format(' '.join(hyponym_value)))
            candidate_hypernym_IDs = hyponyms_with_hypernym_candidates[hyponym_value]
            contexts = bert_based_nn.tokenize_many_text_pairs_for_bert(
                trainset_preparing.generate_context_pairs_for_submission(
                    unseen_hyponym=hyponym_value, occurrences_of_hyponym=occurrences_of_input_hyponyms[hyponym_idx],
                    synsets_with_sense_ids=synsets_from_wordnet, source_senses=source_senses_from_wordnet,
                    inflected_senses=inflected_senses_from_wordnet, checked_synsets=candidate_hypernym_IDs
                ),
                bert_tokenizer,
                pool_=pool
            )
            X = bert_based_nn.create_dataset_for_bert(text_pairs=contexts, seq_len=max_seq_len, batch_size=batch_size)
            assert isinstance(X, tuple)
            assert len(X) == 2
            assert isinstance(X[0], np.ndarray)
            assert isinstance(X[1], np.ndarray)
            print('  {0} data samples;'.format(X[0].shape[0]))
            probabilities = neural_network.predict(X, batch_size=batch_size)
            if num_monte_carlo > 0:
                for _ in range(num_monte_carlo - 1):
                    probabilities += neural_network.predict(X, batch_size=batch_size)
                probabilities /= float(num_monte_carlo)
            probabilities = probabilities.reshape((max(probabilities.shape),))
            del X
            print('  {0} predicted values;'.format(probabilities.shape[0]))
            assert probabilities.shape[0] >= len(contexts)
            best_synsets = list(map(lambda idx: (contexts[idx][2], probabilities[idx]), range(len(contexts))))
            del contexts, probabilities
            best_synsets.sort(key=lambda it: (-it[1], it[0]))
            selected_synset_IDs = list()
            set_of_synset_IDs = set()
            for synset_id, proba in best_synsets:
                if synset_id not in set_of_synset_IDs:
                    set_of_synset_IDs.add(synset_id)
                    selected_synset_IDs.append(synset_id)
                if len(selected_synset_IDs) >= 10:
                    break
            print('  {0} selected synsets.'.format(len(selected_synset_IDs)))
            del best_synsets
            for synset_id in selected_synset_IDs:
                fp.write('{0}\t{1}\t{2}\n'.format(' '.join(hyponym_value).upper(), synset_id,
                                                  synsets_from_wordnet[synset_id][1]))
            del selected_synset_IDs, set_of_synset_IDs
            gc.collect()


def select_input_files(input_dir: str, track_name: str) -> Tuple[str, str]:
    public_suffix = '{0}_public.tsv'.format(track_name.lower())
    private_suffix = '{0}_private.tsv'.format(track_name.lower())
    all_files = list(filter(
        lambda it: it.lower().endswith(public_suffix) or it.lower().endswith(private_suffix),
        os.listdir(input_dir)
    ))
    err = 'Directory `{0}` does not contain input data for public and private submission!'.format(input_dir)
    assert len(all_files) >= 2, err
    assert not (all_files[0].lower().endswith(public_suffix) and all_files[1].lower().endswith(public_suffix)), err
    assert not (all_files[0].lower().endswith(private_suffix) and all_files[1].lower().endswith(private_suffix)), err
    if all_files[0].lower().endswith(public_suffix):
        file_names = (
            os.path.join(input_dir, all_files[0]),
            os.path.join(input_dir, all_files[1])
        )
    else:
        file_names = (
            os.path.join(input_dir, all_files[1]),
            os.path.join(input_dir, all_files[0])
        )
    print('Data for public submission is in the file `{0}`.'.format(file_names[0]))
    print('Data for private submission is in the file `{0}`.'.format(file_names[1]))
    print('')
    return file_names


def main():
    random.seed(142)
    np.random.seed(142)
    tf.random.set_seed(142)
    nltk.download('punkt')

    parser = ArgumentParser()
    parser.add_argument('-t', '--track', dest='track_name', type=str, required=True, choices=['nouns', 'verbs'],
                        help='A competition track name (nouns or verbs).')
    parser.add_argument('-i', '--input', dest='input_data_dir', type=str, required=True,
                        help='A directory with input data, i.e. lists of unseen hyponyms for public and private '
                             'submission.')
    parser.add_argument('-o', '--output', dest='output_data_dir', type=str, required=True,
                        help='A directory with output data, i.e. lists of unseen hyponyms and their hypernyms, found '
                             'as a result of this program execution, for public and private submission.')
    parser.add_argument('-w', '--wordnet', dest='wordnet_dir', type=str, required=True,
                        help='A directory with unarchived RuWordNet.')
    parser.add_argument('-c', '--cache_dir', dest='cache_dir', type=str, required=True,
                        help='A directory with cached data for training.')
    parser.add_argument('--bert', dest='bert_model_dir', type=str, required=False, default=None,
                        help='A directory with pre-trained BERT model.')
    parser.add_argument('--filters', dest='filters_number', type=int, required=False, default=200,
                        help='A number of output filters in each convolution layer.')
    parser.add_argument('--hidden', dest='hidden_layer_size', type=int, required=False, default=2000,
                        help='A hidden layer size.')
    parser.add_argument('--lr', dest='learning_rate', type=float, required=False, default=1e-5, help='A learning rate.')
    parser.add_argument('--epochs', dest='max_epochs', type=int, required=False, default=10,
                        help='A maximal number of training epochs.')
    parser.add_argument('--batch', dest='batch_size', type=int, required=False, default=32, help='A mini-batch size.')
    parser.add_argument('--pooling', dest='pooling_type', type=str, required=False, default='max',
                        choices=['max', 'maximum', 'maximal', 'ave', 'average'],
                        help='A pooling type (`max` or `ave`).')
    parser.add_argument('--nn_head', dest='nn_head_type', type=str, required=False, default='simple',
                        choices=['simple', 'cnn', 'bayesian_cnn'],
                        help='A type of neural network\'s head after BERT (`simple`, `cnn` or `bayesian_cnn`).')
    parser.add_argument('--monte_carlo', dest='num_monte_carlo', type=int, required=False, default=10,
                        help='A sample number for the Monte Carlo inference in a bayesian neural network.')
    parser.add_argument('--kl_weight', dest='kl_weight', type=float, required=False, default=0.01,
                        help='Weight of the KL loss for Bayesian deep learning.')
    args = parser.parse_args()

    cached_data_dir = os.path.normpath(args.cache_dir)
    assert os.path.isdir(cached_data_dir), 'Directory `{0}` does not exist!'.format(cached_data_dir)

    assert args.batch_size > 0, 'A mini-batch size must be a positive value!'
    num_monte_carlo = args.num_monte_carlo if args.nn_head_type == 'bayesian_cnn' else 0
    if num_monte_carlo != 0:
        assert num_monte_carlo > 1, 'A sample number for the Monte Carlo inference must be greater than 1.'
    kl_weight = args.kl_weight
    assert (kl_weight > 0.0) and (kl_weight <= 1.0)

    input_data_dir = os.path.normpath(args.input_data_dir)
    os.path.isdir(input_data_dir), 'Directory `{0}` does not exist!'.format(input_data_dir)
    output_data_dir = os.path.normpath(args.output_data_dir)
    os.path.isdir(output_data_dir), 'Directory `{0}` does not exist!'.format(output_data_dir)
    public_data_name, private_data_name = select_input_files(input_data_dir, args.track_name)
    assert os.path.isfile(public_data_name), 'File `{0}` does not exist!'.format(public_data_name)
    assert os.path.isfile(private_data_name), 'File `{0}` does not exist!'.format(private_data_name)
    public_submission_name = os.path.join(output_data_dir,
                                          'submitted_{0}_public.tsv'.format(('nouns' if args.track_name == 'nouns'
                                                                             else 'verbs')))
    private_submission_name = os.path.join(output_data_dir,
                                           'submitted_{0}_private.tsv'.format(('nouns' if args.track_name == 'nouns'
                                                                               else 'verbs')))
    data_for_public_submission = hyponyms_loading.load_submission_result(public_data_name)
    print('Number of hyponyms for public submission is {0}.'.format(len(data_for_public_submission)))
    data_for_private_submission = hyponyms_loading.load_submission_result(private_data_name)
    print('Number of hyponyms for private submission is {0}.'.format(len(data_for_private_submission)))
    print('')

    file_name = os.path.join(cached_data_dir, 'submission_occurrences_in_texts.json')
    assert os.path.isfile(file_name), 'File `{0}` does not exist!'.format(file_name)
    all_submission_occurrences = text_processing.load_sense_occurrences_in_texts(file_name)
    term_occurrences_for_public = dict()
    term_occurrences_for_private = dict()
    n_public = len(data_for_public_submission)
    n_private = len(data_for_private_submission)
    for idx in range(n_public):
        term_id = str(idx)
        term_occurrences_for_public[term_id] = all_submission_occurrences[term_id]
    assert len(term_occurrences_for_public) == len(data_for_public_submission)
    print('Occurrences of {0} terms from the public submission set have been loaded.'.format(
        len(term_occurrences_for_public)))
    for idx in range(n_public, n_public + n_private):
        term_id = str(idx)
        term_occurrences_for_private[term_id] = all_submission_occurrences[term_id]
    assert len(term_occurrences_for_private) == len(data_for_private_submission)
    print('Occurrences of {0} terms from the private submission set have been loaded.'.format(
        len(term_occurrences_for_private)))
    del all_submission_occurrences
    print('')

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
    synsets, source_senses = ruwordnet_parsing.load_synsets_with_sense_IDs(senses_file_name=wordnet_senses_name,
                                                                           synsets_file_name=wordnet_synsets_name)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        inflected_senses = ruwordnet_parsing.load_and_inflect_senses(
            senses_file_name=wordnet_senses_name,
            main_pos_tag="NOUN" if args.track_name == 'nouns' else "VERB"
        )
    print('')

    if args.nn_head_type == 'simple':
        solver_name = os.path.join(cached_data_dir, 'simple_bert_nn.h5')
        solver_params_name = os.path.join(cached_data_dir, 'simple_bert_params.pkl')
    else:
        solver_name = os.path.join(cached_data_dir, 'bert_and_cnn.h5')
        solver_params_name = os.path.join(cached_data_dir, 'params_of_bert_and_cnn.pkl')
    if os.path.isdir(solver_name) and os.path.isfile(solver_params_name):
        with open(solver_params_name, 'rb') as fp:
            optimal_seq_len, tokenizer = pickle.load(fp)
        assert (optimal_seq_len > 0) and (optimal_seq_len <= bert_based_nn.MAX_SEQ_LENGTH)
        with tf.keras.utils.custom_object_scope({'RAdam': RAdam}):
            solver = tf.keras.models.load_model(solver_name)
        print('The neural network has been loaded from the `{0}`...'.format(solver_name))
    else:
        assert args.bert_model_dir is not None, 'A directory with pre-trained BERT model is not specified!'
        bert_model_dir = os.path.normpath(args.bert_model_dir)
        assert os.path.isdir(bert_model_dir), 'The directory `{0}` does not exist!'.format(bert_model_dir)
        tokenizer = bert_based_nn.initialize_tokenizer(bert_model_dir)
        tokenized_data_name = os.path.join(cached_data_dir, 'tokenized_data_for_BERT.pkl')
        if os.path.isfile(tokenized_data_name):
            with open(tokenized_data_name, 'rb') as fp:
                data_for_training, data_for_validation, data_for_testing = pickle.load(fp)
        else:
            file_name = os.path.join(cached_data_dir, 'contexts_for_training.csv')
            data_for_training = bert_based_nn.tokenize_many_text_pairs_for_bert(
                trainset_preparing.load_context_pairs_from_csv(file_name),
                tokenizer
            )
            print('Number of samples for training is {0}.'.format(len(data_for_training)))
            file_name = os.path.join(cached_data_dir, 'contexts_for_validation.csv')
            data_for_validation = bert_based_nn.tokenize_many_text_pairs_for_bert(
                trainset_preparing.load_context_pairs_from_csv(file_name),
                tokenizer
            )
            print('Number of samples for validation is {0}.'.format(len(data_for_validation)))
            file_name = os.path.join(cached_data_dir, 'contexts_for_testing.csv')
            data_for_testing = bert_based_nn.tokenize_many_text_pairs_for_bert(
                trainset_preparing.load_context_pairs_from_csv(file_name),
                tokenizer
            )
            print('Number of samples for final testing is {0}.'.format(len(data_for_testing)))
            print('')
            with open(tokenized_data_name, 'wb') as fp:
                pickle.dump((data_for_training, data_for_validation, data_for_testing), fp,
                            protocol=pickle.HIGHEST_PROTOCOL)
        lengths_of_texts = [len(it[0]) for it in data_for_training] + [len(it[0]) for it in data_for_validation] + \
                           [len(it[0]) for it in data_for_testing]
        optimal_seq_len = bert_based_nn.calculate_optimal_number_of_tokens(lengths_of_texts)
        del lengths_of_texts
        print('')
        if optimal_seq_len < bert_based_nn.MAX_SEQ_LENGTH:
            data_for_training = list(filter(lambda it: len(it[0]) <= optimal_seq_len, data_for_training))
            data_for_validation = list(filter(lambda it: len(it[0]) <= optimal_seq_len, data_for_validation))
            data_for_testing = list(filter(lambda it: len(it[0]) <= optimal_seq_len, data_for_testing))
            print('Number of filtered samples for training is {0}.'.format(len(data_for_training)))
            print('Number of filtered samples for validation is {0}.'.format(len(data_for_validation)))
            print('Number of filtered samples for final testing is {0}.'.format(len(data_for_testing)))
            print('')
        gc.collect()
        if args.nn_head_type == 'simple':
            solver = bert_based_nn.build_simple_bert(bert_model_dir, max_seq_len=optimal_seq_len,
                                                     learning_rate=args.learning_rate)
        else:
            solver = bert_based_nn.build_bert_and_cnn(
                bert_model_dir, n_filters=args.filters_number, hidden_layer_size=args.hidden_layer_size,
                optimal_seq_len=optimal_seq_len, kl_weight=kl_weight / float(args.batch_size),
                bayesian=(args.nn_head_type == 'bayesian_cnn'), ave_pooling=args.pooling_type in {'ave', 'average'},
                learning_rate=args.learning_rate, max_seq_len=optimal_seq_len
            )

        trainset_generator = bert_based_nn.TrainsetGenerator(text_pairs=data_for_training, seq_len=optimal_seq_len,
                                                             batch_size=args.batch_size)
        del data_for_training
        print('Number of batches for training is {0}.'.format(len(trainset_generator)))
        validset_generator = bert_based_nn.TrainsetGenerator(text_pairs=data_for_validation, seq_len=optimal_seq_len,
                                                             batch_size=args.batch_size)
        del data_for_validation
        print('Number of batches for validation is {0}.'.format(len(validset_generator)))
        testset_generator = bert_based_nn.TrainsetGenerator(text_pairs=data_for_testing, seq_len=optimal_seq_len,
                                                            batch_size=args.batch_size)
        del data_for_testing
        print('Number of batches for final testing is {0}.'.format(len(testset_generator)))
        print('')

        solver = bert_based_nn.train_neural_network(
            trainset_generator=trainset_generator, validset_generator=validset_generator,
            neural_network=solver, max_epochs=args.max_epochs, bayesian=(args.nn_head_type == 'bayesian_cnn')
        )
        del trainset_generator, validset_generator
        gc.collect()
        bert_based_nn.evaluate_neural_network(testset_generator=testset_generator, neural_network=solver,
                                              num_monte_carlo=num_monte_carlo)
        del testset_generator
        solver.save(solver_name)
        with open(solver_params_name, 'wb') as fp:
            pickle.dump((optimal_seq_len, tokenizer), fp)
        gc.collect()
    print('')

    print('Public submission is started...')
    do_submission(
        submission_result_name=public_submission_name,
        input_hyponyms=data_for_public_submission, occurrences_of_input_hyponyms=term_occurrences_for_public,
        synsets_from_wordnet=synsets, source_senses_from_wordnet=source_senses,
        inflected_senses_from_wordnet=inflected_senses, bert_tokenizer=tokenizer,
        neural_network=solver, max_seq_len=optimal_seq_len, batch_size=args.batch_size, num_monte_carlo=num_monte_carlo
    )
    print('Public submission is finished...')
    print('')
    print('Private submission is started...')
    do_submission(
        submission_result_name=private_submission_name,
        input_hyponyms=data_for_private_submission, occurrences_of_input_hyponyms=term_occurrences_for_private,
        synsets_from_wordnet=synsets, source_senses_from_wordnet=source_senses,
        inflected_senses_from_wordnet=inflected_senses, bert_tokenizer=tokenizer,
        neural_network=solver, max_seq_len=optimal_seq_len, batch_size=args.batch_size, num_monte_carlo=num_monte_carlo
    )
    print('Private submission is finished...')
    print('')


if __name__ == '__main__':
    main()
