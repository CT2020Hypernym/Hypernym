from argparse import ArgumentParser
import gzip
import multiprocessing
import os
import pickle
import random
import warnings

import nltk
import numpy as np

from hyponyms_loading import load_terms_for_submission
from ruwordnet_parsing import load_and_inflect_senses, load_synsets_with_sense_IDs
from text_processing import load_sense_occurrences_in_texts
from trainset_preparing import generate_context_pairs_for_submission
from bert_based_nn import tokenize_many_text_pairs_for_bert, initialize_tokenizer


def main():
    random.seed(142)
    nltk.download('punkt')

    parser = ArgumentParser()
    parser.add_argument('-o', '--occ', dest='submission_occurrences_file', required=True, type=str,
                        help='The JSON file with found contexts of unseen hyponyms from public and private sub-sets.')
    parser.add_argument('-w', '--wordnet', dest='wordnet_dir', type=str, required=True,
                        help='A directory with unarchived RuWordNet.')
    parser.add_argument('-t', '--track', dest='track_name', type=str, required=True, choices=['nouns', 'verbs'],
                        help='A competition track name (nouns or verbs).')
    parser.add_argument('-d', '--dst', dest='destination_dir', type=str, required=True,
                        help='A destination directory into which all results (as CSV files) will be saved.')
    parser.add_argument('-b', '--public', dest='public_data', type=str, required=True,
                        help='A text file with a list of unseen hyponyms for public submission.')
    parser.add_argument('-r', '--private', dest='private_data', type=str, required=True,
                        help='A text file with a list of unseen hyponyms for private submission.')
    parser.add_argument('--bert', dest='bert_model_dir', type=str, required=False, default=None,
                        help='A directory with pre-trained BERT model.')
    args = parser.parse_args()

    destination_dir = os.path.normpath(args.destination_dir)
    assert os.path.isdir(destination_dir), 'Directory `{0}` does not exist!'.format(destination_dir)
    if not os.path.isdir(os.path.join(destination_dir, 'context_pairs_for_public')):
        os.mkdir(os.path.join(destination_dir, 'context_pairs_for_public'))
    if not os.path.isdir(os.path.join(destination_dir, 'context_pairs_for_private')):
        os.mkdir(os.path.join(destination_dir, 'context_pairs_for_private'))

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
    synsets, source_senses = load_synsets_with_sense_IDs(senses_file_name=wordnet_senses_name,
                                                         synsets_file_name=wordnet_synsets_name)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        inflected_senses = load_and_inflect_senses(senses_file_name=wordnet_senses_name,
                                                   main_pos_tag="NOUN" if args.track_name == 'nouns' else "VERB")
    print('')

    public_data_name = os.path.normpath(args.public_data)
    assert os.path.isfile(public_data_name), 'File `{0}` does not exist!'.format(public_data_name)
    private_data_name = os.path.normpath(args.private_data)
    assert os.path.isfile(private_data_name), 'File `{0}` does not exist!'.format(private_data_name)
    data_for_public_submission = load_terms_for_submission(public_data_name)
    print('Number of hyponyms for public submission is {0}.'.format(len(data_for_public_submission)))
    data_for_private_submission = load_terms_for_submission(private_data_name)
    print('Number of hyponyms for private submission is {0}.'.format(len(data_for_private_submission)))
    print('')

    file_name = os.path.normpath(args.submission_occurrences_file)
    assert os.path.isfile(file_name), 'File `{0}` does not exist!'.format(file_name)
    all_submission_occurrences = load_sense_occurrences_in_texts(file_name)
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

    assert args.bert_model_dir is not None, 'A directory with pre-trained BERT model is not specified!'
    bert_model_dir = os.path.normpath(args.bert_model_dir)
    assert os.path.isdir(bert_model_dir), 'The directory `{0}` does not exist!'.format(bert_model_dir)
    tokenizer = initialize_tokenizer(bert_model_dir)
    print('The BERT tokenizer has been initialized...')
    print('')

    n_processes = os.cpu_count()
    if n_processes > 1:
        pool = multiprocessing.Pool(processes=n_processes)
    else:
        pool = None

    n_data_parts = 8
    data_part_size = int(np.ceil(len(data_for_public_submission) / float(n_data_parts)))
    start_pos = 0
    data = []
    print('Data preparing for public submission is started...')
    for hyponym_idx, hyponym_value in enumerate(data_for_public_submission):
        print('Unseen hyponym `{0}`:'.format(' '.join(hyponym_value)))
        contexts = tokenize_many_text_pairs_for_bert(
            generate_context_pairs_for_submission(
                unseen_hyponym=hyponym_value, occurrences_of_hyponym=term_occurrences_for_public[hyponym_idx],
                synsets_with_sense_ids=synsets, source_senses=source_senses,
                inflected_senses=inflected_senses
            ),
            tokenizer,
            pool_=pool
        )
        print('  {0} context pairs.'.format(len(contexts)))
        data.append(tuple(contexts))
        del contexts
        if len(data) >= data_part_size:
            data_file_name = os.path.join(
                destination_dir,
                'context_pairs_for_public_{0}_{1}.pkl'.format(start_pos, len(data) + start_pos)
            )
            with open(data_file_name, 'wb') as fp:
                pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
            start_pos += len(data)
            data.clear()
    if len(data) > 0:
        data_file_name = os.path.join(
            destination_dir,
            'context_pairs_for_private_{0}_{1}.pkl'.format(start_pos, len(data) + start_pos)
        )
        with open(data_file_name, 'wb') as fp:
            pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
        start_pos += len(data)
        data.clear()
    print('Data preparing for public submission is finished...')
    print('')
    del data

    data_part_size = int(np.ceil(len(data_for_public_submission) / float(n_data_parts)))
    start_pos = 0
    data = []
    print('Data preparing for private submission is started...')
    for hyponym_idx, hyponym_value in enumerate(data_for_private_submission):
        print('Unseen hyponym `{0}`:'.format(' '.join(hyponym_value)))
        contexts = tokenize_many_text_pairs_for_bert(
            generate_context_pairs_for_submission(
                unseen_hyponym=hyponym_value, occurrences_of_hyponym=term_occurrences_for_private[hyponym_idx],
                synsets_with_sense_ids=synsets, source_senses=source_senses,
                inflected_senses=inflected_senses
            ),
            tokenizer,
            pool_=pool
        )
        print('  {0} context pairs.'.format(len(contexts)))
        del contexts
        if len(data) >= data_part_size:
            data_file_name = os.path.join(
                destination_dir,
                'context_pairs_for_private_{0}_{1}.pkl'.format(start_pos, len(data) + start_pos)
            )
            with open(data_file_name, 'wb') as fp:
                pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
            start_pos += len(data)
            data.clear()
    if len(data) > 0:
        data_file_name = os.path.join(
            destination_dir,
            'context_pairs_for_private_{0}_{1}.pkl'.format(start_pos, len(data) + start_pos)
        )
        with open(data_file_name, 'wb') as fp:
            pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
        start_pos += len(data)
        data.clear()
    print('Data preparing for private submission is finished...')
    print('')
    del data
    print('Data preparing for private submission is finished...')


if __name__ == '__main__':
    main()
