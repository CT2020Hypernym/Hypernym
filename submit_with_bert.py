from argparse import ArgumentParser
import os
import pickle
import random
import warnings

import nltk
import numpy as np
import tensorflow as tf

import ruwordnet_parsing
import trainset_preparing
import hyponyms_loading
import bert_based_nn
import text_processing


def main():
    random.seed(142)
    np.random.seed(142)
    tf.random.set_seed(142)
    nltk.download('punkt')

    parser = ArgumentParser()
    parser.add_argument('-t', '--track', dest='track_name', type=str, required=True, choices=['nouns', 'verbs'],
                        help='A competition track name (nouns or verbs).')
    parser.add_argument('-w', '--wordnet', dest='wordnet_dir', type=str, required=True,
                        help='A directory with unarchived RuWordNet.')
    parser.add_argument('-b', '--public', dest='public_data', type=str, required=True,
                        help='A text file with a list of unseen hyponyms for public submission.')
    parser.add_argument('-r', '--private', dest='private_data', type=str, required=True,
                        help='A text file with a list of unseen hyponyms for private submission.')
    parser.add_argument('-c', '--cache_dir', dest='cache_dir', type=str, required=True,
                        help='A directory with cached data for training.')
    parser.add_argument('--filters', dest='filters_number', type=int, required=False, default=200,
                        help='A number of output filters in each convolution layer.')
    parser.add_argument('--hidden', dest='hidden_layer_size', type=int, required=False, default=2000,
                        help='A hidden layer size.')
    parser.add_argument('--lr_max', dest='max_learning_rate', type=float, required=False, default=1e-3,
                        help='A maximal learning rate for the cyclical learning rate schedule.')
    parser.add_argument('--lr_min', dest='min_learning_rate', type=float, required=False, default=1e-5,
                        help='A minimal learning rate for the cyclical learning rate schedule.')
    parser.add_argument('--cycle_length', dest='training_cycle_length', type=int, required=False, default=1500,
                        help='A period of cycle length for the cyclical learning rate schedule (in mini-batches).')
    parser.add_argument('--iters', dest='max_iters', type=int, required=False, default=100000,
                        help='A maximal number of iterations (in mini-batches) to train the neural network.')
    parser.add_argument('--eval_every', dest='eval_every', type=int, required=False, default=1000,
                        help='Number of iterations (in mini-batches) between evaluation on the validation subset.')
    parser.add_argument('--batch', dest='batch_size', type=int, required=False, default=64, help='A mini-batch size.')
    parser.add_argument('--bert', dest='bert_model_name', type=str, required=False,
                        default='http://files.deeppavlov.ai/deeppavlov_data/bert/'
                                'rubert_cased_L-12_H-768_A-12_v2.tar.gz',
                        help='A pre-trained BERT model.')
    parser.add_argument('--nn_head', dest='nn_head_type', type=str, required=False, default='simple',
                        choices=['simple', 'cnn', 'bayesian_cnn'],
                        help='A type of neural network\'s head after BERT (`simple`, `cnn` or `bayesian_cnn`).')
    args = parser.parse_args()

    cached_data_dir = os.path.normpath(args.cache_dir)
    assert os.path.isdir(cached_data_dir), 'Directory `{0}` does not exist!'.format(cached_data_dir)

    assert args.nn_head_type != 'bayesian_cnn', 'A Bayesian neural network has not implemented yet.'

    wordnet_dir = os.path.normpath(args.wordnet_dir)
    assert os.path.isdir(wordnet_dir)
    wordnet_senses_name = os.path.join(wordnet_dir, 'senses.N.xml' if args.track_name == 'nouns' else 'senses.V.xml')
    wordnet_synsets_name = os.path.join(wordnet_dir, 'synsets.N.xml' if args.track_name == 'nouns' else 'synsets.V.xml')
    wordnet_relations_name = os.path.join(
        wordnet_dir,
        'synset_relations.N.xml' if args.track_name == 'nouns' else 'synset_relations.V.xml'
    )
    assert os.path.isfile(wordnet_senses_name), 'File `{0}` does not exist!'.format(wordnet_senses_name)
    assert os.path.isfile(wordnet_synsets_name), 'File `{0}` does not exist!'.format(wordnet_synsets_name)
    assert os.path.isfile(wordnet_relations_name), 'File `{0}` does not exist!'.format(wordnet_relations_name)
    synsets, source_senses = ruwordnet_parsing.load_synsets_with_sense_IDs(senses_file_name=wordnet_senses_name,
                                                                           synsets_file_name=wordnet_synsets_name)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        inflected_senses = ruwordnet_parsing.load_and_inflect_senses(
            senses_file_name=wordnet_senses_name,
            main_pos_tag="NOUN" if args.track_name == 'nouns' else "VERB"
        )
    print('')

    public_data_name = os.path.normpath(args.public_data)
    assert os.path.isfile(public_data_name), 'File `{0}` does not exist!'.format(public_data_name)
    public_submission_name = os.path.join(os.path.dirname(public_data_name),
                                          'submitted_' + os.path.basename(public_data_name))
    private_data_name = os.path.normpath(args.private_data)
    assert os.path.isfile(private_data_name), 'File `{0}` does not exist!'.format(private_data_name)
    private_submission_name = os.path.join(os.path.dirname(private_data_name),
                                           'submitted_' + os.path.basename(private_data_name))
    data_for_public_submission = hyponyms_loading.load_terms_for_submission(public_data_name)
    print('Number of hyponyms for public submission is {0}.'.format(len(data_for_public_submission)))
    data_for_private_submission = hyponyms_loading.load_terms_for_submission(private_data_name)
    print('Number of hyponyms for private submission is {0}.'.format(len(data_for_private_submission)))
    print('')

    file_name = os.path.join(cached_data_dir, 'submission_occurrences_in_texts.json')
    assert os.path.isfile(file_name), 'File `{0}` does not exist!'.format(file_name)
    all_submission_occurrences = text_processing.load_sense_occurrences_in_texts(file_name)
    term_occurrences_for_public = []
    term_occurrences_for_private = []
    n_public = len(data_for_public_submission)
    n_private = len(data_for_private_submission)
    for idx in range(n_public):
        term_id = str(idx)
        term_occurrences_for_public.append(all_submission_occurrences[term_id])
    assert len(term_occurrences_for_public) == len(data_for_public_submission)
    print('Occurrences of {0} terms from the public submission set have been loaded.'.format(
        len(term_occurrences_for_public)))
    for idx in range(n_public, n_public + n_private):
        term_id = str(idx)
        term_occurrences_for_private.append(all_submission_occurrences[term_id])
    assert len(term_occurrences_for_private) == len(data_for_private_submission)
    print('Occurrences of {0} terms from the private submission set have been loaded.'.format(
        len(term_occurrences_for_private)))
    del all_submission_occurrences
    print('')

    if args.nn_head_type == 'simple':
        tokenizer, solver = bert_based_nn.build_simple_bert(args.bert_model_name)
        solver_name = os.path.join(cached_data_dir, 'simple_bert_nn.h5py')
        solver_params_name = os.path.join(cached_data_dir, 'simple_bert_params.pkl')
    else:
        tokenizer, solver = bert_based_nn.build_bert_and_cnn(args.bert_model_name, n_filters=args.filters_number,
                                                             hidden_layer_size=args.hidden_layer_size)
        solver_name = os.path.join(cached_data_dir, 'bert_and_cnn.h5py')
        solver_params_name = os.path.join(cached_data_dir, 'params_of_bert_and_cnn.pkl')
    print('The neural network has been built...')
    print('')
    if os.path.isfile(solver_name) and os.path.isfile(solver_params_name):
        with open(solver_params_name, 'rb') as fp:
            optimal_seq_len = pickle.load(fp)
        assert (optimal_seq_len > 0) and (optimal_seq_len <= bert_based_nn.MAX_SEQ_LENGTH)
        if optimal_seq_len < bert_based_nn.MAX_SEQ_LENGTH:
            del solver, tokenizer
            tf.keras.backend.clear_session()
            if args.nn_head_type == 'simple':
                tokenizer, solver = bert_based_nn.build_simple_bert(args.bert_model_name,
                                                                    optimal_seq_len=optimal_seq_len)
            else:
                tokenizer, solver = bert_based_nn.build_bert_and_cnn(args.bert_model_name,
                                                                     n_filters=args.filters_number,
                                                                     hidden_layer_size=args.hidden_layer_size,
                                                                     optimal_seq_len=optimal_seq_len)
        solver.load_weights(solver_name)
        print('The neural network has been loaded from file `{0}`...'.format(solver_name))
    else:
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
            del solver, tokenizer
            tf.keras.backend.clear_session()
            if args.nn_head_type == 'simple':
                tokenizer, solver = bert_based_nn.build_simple_bert(args.bert_model_name,
                                                                    optimal_seq_len=optimal_seq_len)
            else:
                tokenizer, solver = bert_based_nn.build_bert_and_cnn(args.bert_model_name,
                                                                     n_filters=args.filters_number,
                                                                     hidden_layer_size=args.hidden_layer_size,
                                                                     optimal_seq_len=optimal_seq_len)

        trainset_generator = bert_based_nn.BertDatasetGenerator(
            text_pairs=data_for_training, batch_size=args.batch_size, seq_len=optimal_seq_len
        )
        print('Number of mini-batches for training is {0}.'.format(len(trainset_generator)))
        validset_generator = bert_based_nn.BertDatasetGenerator(
            text_pairs=data_for_validation, batch_size=args.batch_size, seq_len=optimal_seq_len
        )
        print('Number of mini-batches for validation is {0}.'.format(len(validset_generator)))
        testset_generator = bert_based_nn.BertDatasetGenerator(
            text_pairs=data_for_testing, batch_size=args.batch_size, seq_len=optimal_seq_len
        )
        print('Number of mini-batches for final testing is {0}.'.format(len(testset_generator)))
        del data_for_training, data_for_validation, data_for_testing

        solver = bert_based_nn.train_neural_network(
            data_for_training=trainset_generator, data_for_validation=validset_generator,
            neural_network=solver, max_iters=args.max_iters, training_cycle_length=args.training_cycle_length,
            eval_every=args.eval_every, max_learning_rate=args.max_learning_rate,
            min_learning_rate=args.min_learning_rate,
            is_bayesian=False
        )
        del trainset_generator, validset_generator
        bert_based_nn.evaluate_neural_network(dataset=testset_generator, neural_network=solver, num_monte_carlo=0)
        del testset_generator
        solver.save_weights(solver_name)
        with open(solver_params_name, 'wb') as fp:
            pickle.dump(optimal_seq_len, fp)
    print('')

    print('Public submission is started...')
    bert_based_nn.do_submission(
        submission_result_name=public_submission_name,
        neural_network=solver, bert_tokenizer=tokenizer, max_seq_len=optimal_seq_len, batch_size=args.batch_size,
        input_hyponyms=data_for_public_submission, occurrences_of_input_hyponyms=term_occurrences_for_public,
        wordnet_synsets=synsets, wordnet_source_senses=source_senses, wordnet_inflected_senses=inflected_senses
    )
    print('Public submission is finished...')
    print('')
    print('Private submission is started...')
    bert_based_nn.do_submission(
        submission_result_name=private_submission_name,
        neural_network=solver, bert_tokenizer=tokenizer, max_seq_len=optimal_seq_len, batch_size=args.batch_size,
        input_hyponyms=data_for_private_submission, occurrences_of_input_hyponyms=term_occurrences_for_private,
        wordnet_synsets=synsets, wordnet_source_senses=source_senses, wordnet_inflected_senses=inflected_senses
    )
    print('Private submission is finished...')
    print('')


if __name__ == '__main__':
    main()
