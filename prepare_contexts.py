from argparse import ArgumentParser
import codecs
from datetime import datetime
import json
import os
import random

import nltk
import numpy as np

from ruwordnet_parsing import load_and_inflect_senses
from text_processing import load_news, load_wiki, update_sense_entries_in_texts, prepare_senses_index_for_search


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
                        help='The JSON file with found contexts of the RuWordNet terms (senses).')
    parser.add_argument('-w', '--wordnet', dest='wordnet_dir', type=str, required=True,
                        help='A directory with unarchived RuWordNet.')
    parser.add_argument('-t', '--track', dest='track_name', type=str, required=True, choices=['nouns', 'verbs'],
                        help='A competition track name (nouns or verbs).')
    args = parser.parse_args()

    nltk.download('punkt')
    wordnet_dir = os.path.normpath(args.wordnet_dir)
    assert os.path.isdir(wordnet_dir)
    wordnet_senses_name = os.path.join(wordnet_dir, 'senses.N.xml' if args.track_name == 'nouns' else 'senses.V.xml')
    assert args.data_source != "librusec", "The processing of the LibRuSec text corpus is not implemented already!"

    full_path = os.path.normpath(args.source_path)
    if args.data_source == "news":
        assert os.path.isdir(full_path), 'The directory "{0}" does not exist!'.format(full_path)
    else:
        assert os.path.isfile(full_path), 'The file "{0}" does not exist!'.format(full_path)

    result_file_name = os.path.normpath(args.json_file)
    result_file_dir = os.path.dirname(result_file_name)
    if len(result_file_dir) > 0:
        assert os.path.isdir(full_path), 'The directory "{0}" does not exist!'.format(result_file_dir)
    assert not os.path.isdir(result_file_name), '"{0}" is a directory, but a file is expected.'.format(result_file_name)

    senses = load_and_inflect_senses(wordnet_senses_name, "NOUN" if args.track_name == 'nouns' else "VERB")
    print("")
    search_index = prepare_senses_index_for_search(senses)

    all_texts = []
    all_entries_of_senses = dict()
    generator = load_news(full_path) if args.data_source == "news" else load_wiki(full_path)
    counter = 1
    for new_text in generator:
        update_sense_entries_in_texts(new_text, senses_dict=senses, search_index_for_senses=search_index,
                                      all_texts=all_texts, all_entries=all_entries_of_senses)
        if counter % 10000 == 0:
            print("{0}: {1} texts have been processed.".format(
                datetime.now().strftime("%A, %d %B %Y, %I:%M %p"), counter
            ))
            print('  {0} terms (senses) from {1} have been found in {2} texts.'.format(len(all_entries_of_senses),
                                                                                       len(senses), len(all_texts)))
        counter += 1
    if (counter - 1) % 10000 != 0:
        print("{0}: {1} texts have been processed.".format(
            datetime.now().strftime("%A, %d %B %Y, %I:%M:%S %p"), counter
        ))
        print('  {0} terms (senses) from {1} have been found in {2} texts.'.format(len(all_entries_of_senses),
                                                                                   len(senses), len(all_texts)))
    if os.path.isfile(result_file_name):
        with codecs.open(filename=result_file_name, mode="r", encoding="utf-8", errors="ignore") as fp:
            results_for_senses = json.load(fp)
    else:
        results_for_senses = dict()
    for sense_id in all_entries_of_senses:
        if sense_id not in results_for_senses:
            results_for_senses[sense_id] = dict()
        for morphotag in all_entries_of_senses[sense_id]:
            if morphotag not in results_for_senses[sense_id]:
                results_for_senses[sense_id][morphotag] = []
            for text_id, entry_bounds in all_entries_of_senses[sense_id][morphotag]:
                results_for_senses[sense_id][morphotag].append({"text": ' '.join(all_texts[text_id]),
                                                                "bounds": entry_bounds})
    with codecs.open(filename=result_file_name, mode="w", encoding="utf-8", errors="ignore") as fp:
        json.dump(results_for_senses, fp, ensure_ascii=False, indent=4, sort_keys=True)


if __name__ == '__main__':
    main()
