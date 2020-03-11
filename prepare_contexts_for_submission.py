from argparse import ArgumentParser
import codecs
from datetime import datetime
import json
import multiprocessing
import os
import random

import nltk
import numpy as np

from hyponyms_loading import load_terms_for_submission, inflect_terms_for_submission
from text_processing import load_news, load_wiki, prepare_senses_index_for_search
from text_processing import calculate_sense_occurrences_in_texts, join_sense_occurrences_in_texts
from text_processing import load_sense_occurrences_in_texts


N_MAX_SENTENCES_PER_MORPHO = 5
MIN_SENTENCE_LENGTH = 7
MAX_SENTENCE_LENGTH = 70


def main():
    random.seed(142)
    np.random.seed(142)

    parser = ArgumentParser()
    parser.add_argument('-d', '--data', dest='data_source', type=str, required=True,
                        choices=['wiki', 'news', 'librusec'],
                        help='A data source kind (wiki, news or librusec), prepared for '
                             'the Taxonomy Enrichment competition.')
    parser.add_argument('-p', '--path', dest='source_path', required=True, type=str,
                        help='Path to the source data file or directory.')
    parser.add_argument('-j', '--json', dest='json_file', type=str, required=True,
                        help='The JSON file with found contexts of terms for submission.')
    parser.add_argument('-b', '--public', dest='public_data', type=str, required=True,
                        help='A text file with a list of unseen hyponyms for public submission.')
    parser.add_argument('-r', '--private', dest='private_data', type=str, required=True,
                        help='A text file with a list of unseen hyponyms for private submission.')
    parser.add_argument('-t', '--track', dest='track_name', type=str, required=True, choices=['nouns', 'verbs'],
                        help='A competition track name (nouns or verbs).')
    args = parser.parse_args()

    nltk.download('punkt')
    public_data_name = os.path.normpath(args.public_data)
    assert os.path.isfile(public_data_name)
    private_data_name = os.path.normpath(args.private_data)
    assert os.path.isfile(private_data_name)
    data_for_public_submission = load_terms_for_submission(public_data_name)
    print('Number of hyponyms for public submission is {0}.'.format(len(data_for_public_submission)))
    data_for_private_submission = load_terms_for_submission(private_data_name)
    print('Number of hyponyms for private submission is {0}.'.format(len(data_for_private_submission)))
    print('')

    full_path = os.path.normpath(args.source_path)
    if args.data_source == "news":
        assert os.path.isdir(full_path), 'The directory "{0}" does not exist!'.format(full_path)
    else:
        assert os.path.isfile(full_path), 'The file "{0}" does not exist!'.format(full_path)

    result_file_name = os.path.normpath(args.json_file)
    result_file_dir = os.path.dirname(result_file_name)
    if len(result_file_dir) > 0:
        assert os.path.isdir(result_file_dir), 'The directory "{0}" does not exist!'.format(result_file_dir)
    assert not os.path.isdir(result_file_name), '"{0}" is a directory, but a file is expected.'.format(result_file_name)

    senses = inflect_terms_for_submission(data_for_public_submission + data_for_private_submission,
                                          "NOUN" if args.track_name == 'nouns' else "VERB")
    print("All terms for submission have been inflected using the PyMorphy2.")
    print("")
    search_index = prepare_senses_index_for_search(senses)

    if os.path.isfile(result_file_name):
        all_occurrences_of_senses = load_sense_occurrences_in_texts(result_file_name)
    else:
        all_occurrences_of_senses = dict()
    generator = load_news(full_path) if args.data_source == "news" else load_wiki(full_path)
    counter = 0
    n_processes = os.cpu_count()
    if n_processes > 1:
        pool = multiprocessing.Pool(processes=n_processes)
    else:
        pool = None
    max_buffer_size = 30000 * max(1, n_processes)
    buffer = []
    for new_text in generator:
        buffer.append(new_text)
        if len(buffer) >= max_buffer_size:
            if pool is None:
                new_occurrences_of_senses = calculate_sense_occurrences_in_texts(
                    source_texts=buffer, senses_dict=senses, search_index_for_senses=search_index,
                    min_sentence_length=MIN_SENTENCE_LENGTH, max_sentence_length=MAX_SENTENCE_LENGTH,
                    n_sentences_per_morpho=N_MAX_SENTENCES_PER_MORPHO
                )
            else:
                n_data_part = int(np.ceil(len(buffer) / float(n_processes)))
                parts_of_buffer = [(buffer[(idx * n_data_part):((idx + 1) * n_data_part)], senses, search_index,
                                    N_MAX_SENTENCES_PER_MORPHO, MIN_SENTENCE_LENGTH, MAX_SENTENCE_LENGTH)
                                   for idx in range(n_processes - 1)]
                parts_of_buffer.append((buffer[((n_processes - 1) * n_data_part):], senses, search_index,
                                        N_MAX_SENTENCES_PER_MORPHO, MIN_SENTENCE_LENGTH, MAX_SENTENCE_LENGTH))
                parts_of_result = list(pool.starmap(calculate_sense_occurrences_in_texts, parts_of_buffer))
                new_occurrences_of_senses = join_sense_occurrences_in_texts(parts_of_result, N_MAX_SENTENCES_PER_MORPHO)
                del parts_of_buffer, parts_of_result
            all_occurrences_of_senses = join_sense_occurrences_in_texts(
                [all_occurrences_of_senses, new_occurrences_of_senses],
                N_MAX_SENTENCES_PER_MORPHO
            )
            del new_occurrences_of_senses
            counter += len(buffer)
            buffer.clear()
            print("{0}: {1} texts have been processed.".format(
                datetime.now().strftime("%A, %d %B %Y, %I:%M %p"), counter
            ))
            print('  {0} terms (senses) from {1} have been found.'.format(len(all_occurrences_of_senses), len(senses)))
    if len(buffer) > 0:
        if pool is None:
            new_occurrences_of_senses = calculate_sense_occurrences_in_texts(
                source_texts=buffer, senses_dict=senses, search_index_for_senses=search_index,
                min_sentence_length=MIN_SENTENCE_LENGTH, max_sentence_length=MAX_SENTENCE_LENGTH,
                n_sentences_per_morpho=N_MAX_SENTENCES_PER_MORPHO
            )
        else:
            n_data_part = int(np.ceil(len(buffer) / float(n_processes)))
            parts_of_buffer = [(buffer[(idx * n_data_part):((idx + 1) * n_data_part)], senses, search_index,
                                N_MAX_SENTENCES_PER_MORPHO, MIN_SENTENCE_LENGTH, MAX_SENTENCE_LENGTH)
                               for idx in range(n_processes - 1)]
            parts_of_buffer.append((buffer[((n_processes - 1) * n_data_part):], senses, search_index,
                                    N_MAX_SENTENCES_PER_MORPHO, MIN_SENTENCE_LENGTH, MAX_SENTENCE_LENGTH))
            parts_of_result = list(pool.starmap(calculate_sense_occurrences_in_texts, parts_of_buffer))
            new_occurrences_of_senses = join_sense_occurrences_in_texts(parts_of_result, N_MAX_SENTENCES_PER_MORPHO)
        all_occurrences_of_senses = join_sense_occurrences_in_texts(
            [all_occurrences_of_senses, new_occurrences_of_senses],
            N_MAX_SENTENCES_PER_MORPHO
        )
        del new_occurrences_of_senses
        counter += len(buffer)
        print("{0}: {1} texts have been processed.".format(
            datetime.now().strftime("%A, %d %B %Y, %I:%M:%S %p"), counter
        ))
        print('  {0} terms (senses) from {1} have been found.'.format(len(all_occurrences_of_senses), len(senses)))
    with codecs.open(filename=result_file_name, mode="w", encoding="utf-8", errors="ignore") as fp:
        json.dump(all_occurrences_of_senses, fp, ensure_ascii=False, indent=4, sort_keys=True)


if __name__ == '__main__':
    main()
