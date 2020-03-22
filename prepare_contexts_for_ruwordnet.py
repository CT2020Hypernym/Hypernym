"""
This module is a part of system for the automatic enrichment
of a WordNet-like taxonomy.

Copyright 2020 Ivan Bondarenko, Tatiana Batura

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
from datetime import datetime
import gc
import json
import os
import random

import nltk
import numpy as np

from bert_based_nn import initialize_tokenizer
from ruwordnet_parsing import load_and_inflect_senses, load_homonyms
from text_processing import load_news, load_wiki, prepare_senses_index_for_search
from text_processing import calculate_sense_occurrences_in_texts, join_sense_occurrences_in_texts
from text_processing import load_sense_occurrences_in_texts
from trainset_preparing import load_fasttext_model
from udpipe_applying import initialize_udpipe


N_MAX_SENTENCES_PER_MORPHO = 10
MIN_SENTENCE_LENGTH = 7
MAX_SENTENCE_LENGTH = 30


def main():
    random.seed(142)
    np.random.seed(142)

    parser = ArgumentParser()
    parser.add_argument('-d', '--data', dest='data_source', type=str, required=True,
                        choices=['wiki', 'news', 'librusec'],
                        help='A data source kind (wiki, news or librusec), prepared for '
                             'the Taxonomy Enrichment competition.')
    parser.add_argument('-f', '--fasttext', dest='fasttext_name', type=str, required=True,
                        help='A binary file with a Facebook-like FastText model (*.bin).')
    parser.add_argument('-p', '--path', dest='source_path', required=True, type=str,
                        help='Path to the source data file or directory.')
    parser.add_argument('-j', '--json', dest='json_file', type=str, required=True,
                        help='The JSON file with found contexts of the RuWordNet terms (senses).')
    parser.add_argument('-w', '--wordnet', dest='wordnet_dir', type=str, required=True,
                        help='A directory with unarchived RuWordNet.')
    parser.add_argument('-t', '--track', dest='track_name', type=str, required=True, choices=['nouns', 'verbs'],
                        help='A competition track name (nouns or verbs).')
    parser.add_argument('-n', '--number', dest='number', type=int, required=False, default=None,
                        help='A maximal number of processed lines in the text corpus '
                             '(if it is not specified then all lines will be processed).')
    parser.add_argument('-u', '--udpipe', dest='udpipe_model', required=False, type=str, default="ru",
                        help='Language of a used SpaCy-UDPipe model.')
    parser.add_argument('-b', '--bert', dest='bert_model_dir', type=str, required=False, default=None,
                        help='A directory with pre-trained BERT model.')
    args = parser.parse_args()

    nltk.download('punkt')
    wordnet_dir = os.path.normpath(args.wordnet_dir)
    assert os.path.isdir(wordnet_dir)
    wordnet_senses_name = os.path.join(wordnet_dir, 'senses.N.xml' if args.track_name == 'nouns' else 'senses.V.xml')
    wordnet_synsets_name = os.path.join(wordnet_dir, 'synsets.N.xml' if args.track_name == 'nouns' else 'synsets.V.xml')
    assert args.data_source != "librusec", "The processing of the LibRuSec text corpus is not implemented already!"
    max_number_of_lines = args.number
    if max_number_of_lines is not None:
        assert max_number_of_lines > 0, 'A maximal number of processed lines must be a positive value!'

    udpipe_model = initialize_udpipe(args.udpipe_model)

    fasttext_model_path = os.path.normpath(args.fasttext_name)
    assert os.path.isfile(fasttext_model_path), 'File `{0}` does not exist!'.format(fasttext_model_path)

    full_path = os.path.normpath(args.source_path)
    if args.data_source == "news":
        assert os.path.isdir(full_path), 'The directory "{0}" does not exist!'.format(full_path)
    else:
        assert os.path.isfile(full_path), 'The file "{0}" does not exist!'.format(full_path)

    assert args.bert_model_dir is not None, 'A directory with pre-trained BERT model is not specified!'
    bert_model_dir = os.path.normpath(args.bert_model_dir)
    assert os.path.isdir(bert_model_dir), 'The directory `{0}` does not exist!'.format(bert_model_dir)

    result_file_name = os.path.normpath(args.json_file)
    result_file_dir = os.path.dirname(result_file_name)
    if len(result_file_dir) > 0:
        assert os.path.isdir(result_file_dir), 'The directory "{0}" does not exist!'.format(result_file_dir)
    assert not os.path.isdir(result_file_name), '"{0}" is a directory, but a file is expected.'.format(result_file_name)

    senses = load_and_inflect_senses(wordnet_senses_name, "NOUN" if args.track_name == 'nouns' else "VERB")
    print("")
    search_index = prepare_senses_index_for_search(senses)
    fasttext_model = load_fasttext_model(fasttext_model_path)
    print('The FastText model has been loaded...')
    print('')
    homonyms = load_homonyms(synsets_file_name=wordnet_synsets_name, senses_file_name=wordnet_senses_name,
                             fasttext_model=fasttext_model, udpipe_pipeline=udpipe_model)
    del udpipe_model
    gc.collect()
    print('The homomyms dictionary has been built...')
    print('')

    bert_tokenizer = initialize_tokenizer(bert_model_dir)
    print('The BERT tokenizer has been initialized...')
    print('')

    if os.path.isfile(result_file_name):
        all_occurrences_of_senses = load_sense_occurrences_in_texts(result_file_name)
    else:
        all_occurrences_of_senses = dict()
    generator = load_news(full_path) if args.data_source == "news" else load_wiki(full_path)
    counter = 0
    max_buffer_size = 30000
    buffer = []
    for new_text in generator:
        buffer.append(new_text)
        if len(buffer) >= max_buffer_size:
            new_occurrences_of_senses = calculate_sense_occurrences_in_texts(
                source_texts=buffer, senses_dict=senses, search_index_for_senses=search_index,
                min_sentence_length=MIN_SENTENCE_LENGTH, max_sentence_length=MAX_SENTENCE_LENGTH,
                n_sentences_per_morpho=N_MAX_SENTENCES_PER_MORPHO, homonyms=homonyms, fasttext_model=fasttext_model,
                bert_tokenizer=bert_tokenizer
            )
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
            if max_number_of_lines is not None:
                if counter >= max_number_of_lines:
                    break
    if len(buffer) > 0:
        new_occurrences_of_senses = calculate_sense_occurrences_in_texts(
            source_texts=buffer, senses_dict=senses, search_index_for_senses=search_index,
            min_sentence_length=MIN_SENTENCE_LENGTH, max_sentence_length=MAX_SENTENCE_LENGTH,
            n_sentences_per_morpho=N_MAX_SENTENCES_PER_MORPHO, homonyms=homonyms, fasttext_model=fasttext_model,
            bert_tokenizer=bert_tokenizer
        )
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
