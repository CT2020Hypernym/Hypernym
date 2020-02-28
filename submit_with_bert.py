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
import bert_based_nn


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
    parser.add_argument('--lr_max', dest='max_learning_rate', type=float, required=False, default=1e-3,
                        help='A maximal learning rate for the cyclical learning rate schedule.')
    parser.add_argument('--lr_min', dest='min_learning_rate', type=float, required=False, default=1e-6,
                        help='A minimal learning rate for the cyclical learning rate schedule.')
    parser.add_argument('--cycle_length', dest='training_cycle_length', type=int, required=False, default=3000,
                        help='A period of cycle length for the cyclical learning rate schedule (in mini-batches).')
    parser.add_argument('--iters', dest='max_iters', type=int, required=False, default=100000,
                        help='A maximal number of iterations (in mini-batches) to train the neural network.')
    parser.add_argument('--eval_every', dest='eval_every', type=int, required=False, default=100,
                        help='Number of iterations (in mini-batches) between evaluation on the validation subset.')
    parser.add_argument('--batch', dest='batch_size', type=int, required=False, default=8, help='A mini-batch size.')
    parser.add_argument('--bert', dest='bert_model_name', type=str, required=False,
                        default='http://files.deeppavlov.ai/deeppavlov_data/bert/'
                                'rubert_cased_L-12_H-768_A-12_v2.tar.gz',
                        help='A pre-trained BERT model.')
    args = parser.parse_args()

    cached_data_dir = os.path.normpath(args.cache_dir)
    assert os.path.isdir(cached_data_dir), 'Directory `{0}` does not exist!'.format(cached_data_dir)

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

    file_name = os.path.join(cached_data_dir, 'contexts_for_training.csv')
    data_for_training = trainset_preparing.load_context_pairs_from_csv(file_name)
    print('Number of samples for training is {0}.'.format(len(data_for_training)))
    file_name = os.path.join(cached_data_dir, 'contexts_for_validation.csv')
    data_for_validation = trainset_preparing.load_context_pairs_from_csv(file_name)
    print('Number of samples for validation is {0}.'.format(len(data_for_validation)))
    file_name = os.path.join(cached_data_dir, 'contexts_for_testing.csv')
    data_for_testing = trainset_preparing.load_context_pairs_from_csv(file_name)
    print('Number of samples for final testing is {0}.'.format(len(data_for_testing)))

    tokenizer, solver = bert_based_nn.build_simple_bert(args.bert_model_name)

    trainset_generator = bert_based_nn.BertDatasetGenerator(
        text_pairs=data_for_training, batch_size=args.batch_size, tokenizer=tokenizer
    )
    validset_generator = bert_based_nn.BertDatasetGenerator(
        text_pairs=data_for_validation, batch_size=args.batch_size, tokenizer=tokenizer
    )
    testset_generator = bert_based_nn.BertDatasetGenerator(
        text_pairs=data_for_testing, batch_size=args.batch_size, tokenizer=tokenizer
    )
    del data_for_training, data_for_validation, data_for_testing

    solver = bert_based_nn.train_neural_network(
        data_for_training=trainset_generator, data_for_validation=validset_generator,
        neural_network=solver, max_iters=args.max_iters, training_cycle_length=args.training_cycle_length,
        eval_every=args.eval_every, max_learning_rate=args.max_learning_rate, min_learning_rate=args.min_learning_rate,
        is_bayesian=False
    )
    del trainset_generator, validset_generator
    bert_based_nn.evaluate_neural_network(dataset=testset_generator, neural_network=solver, num_monte_carlo=0)
    del testset_generator

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
