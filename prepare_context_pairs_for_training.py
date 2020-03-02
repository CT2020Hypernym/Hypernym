from argparse import ArgumentParser
import os
import random

import nltk

from ruwordnet_parsing import load_and_inflect_senses, load_synsets_with_sense_IDs, prepare_data_for_training
from text_processing import load_sense_occurrences_in_texts
from trainset_preparing import generate_context_pairs_for_training, save_context_pairs_to_csv


def main():
    random.seed(142)
    nltk.download('punkt')

    parser = ArgumentParser()
    parser.add_argument('-o', '--word_occ', dest='wordnet_occurrences_file', required=True, type=str,
                        help='The JSON file with found contexts of the RuWordNet terms (senses).')
    parser.add_argument('-w', '--wordnet', dest='wordnet_dir', type=str, required=True,
                        help='A directory with unarchived RuWordNet.')
    parser.add_argument('-t', '--track', dest='track_name', type=str, required=True, choices=['nouns', 'verbs'],
                        help='A competition track name (nouns or verbs).')
    parser.add_argument('-d', '--dst', dest='destination_dir', type=str, required=True,
                        help='A destination directory into which all results (as CSV files) will be saved.')
    args = parser.parse_args()

    destination_dir = os.path.normpath(args.destination_dir)
    assert os.path.isdir(destination_dir), 'Directory `{0}` does not exist!'.format(destination_dir)

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
    inflected_senses = load_and_inflect_senses(senses_file_name=wordnet_senses_name,
                                               main_pos_tag="NOUN" if args.track_name == 'nouns' else "VERB")
    print('')

    file_name = os.path.normpath(args.wordnet_occurrences_file)
    assert os.path.isfile(file_name), 'File `{0}` does not exist!'.format(file_name)
    ruwordnet_occurrences = load_sense_occurrences_in_texts(file_name)
    synsets_with_occurrences = set()
    n_occurrences = 0
    for sense_id in ruwordnet_occurrences:
        for synset_id in synsets:
            if synset_id not in synsets_with_occurrences:
                if sense_id in synsets[synset_id]:
                    synsets_with_occurrences.add(synset_id)
        for morphotag in ruwordnet_occurrences[sense_id]:
            n_occurrences += len(ruwordnet_occurrences[sense_id][morphotag])
    print('{0} occurrences (contexts) for {1} synsets of RuWordNet from {2} have been loaded...'.format(
        n_occurrences, len(synsets_with_occurrences), len(synsets)))
    print('')
    del synsets_with_occurrences

    data_for_training, data_for_validation, data_for_testing = prepare_data_for_training(
        senses_file_name=wordnet_senses_name, synsets_file_name=wordnet_synsets_name,
        relations_file_name=wordnet_relations_name
    )

    contexts_for_training = generate_context_pairs_for_training(
        data=data_for_training, synsets_with_sense_ids=synsets,
        source_senses=source_senses, inflected_senses=inflected_senses,
        sense_occurrences=ruwordnet_occurrences, all_possible_pairs=True
    )
    file_name = os.path.join(destination_dir, 'contexts_for_training.csv')
    save_context_pairs_to_csv(contexts_for_training, file_name)
    print('{0} text pairs have been generated for training.'.format(len(contexts_for_training)))
    print('')
    del contexts_for_training

    contexts_for_validation = generate_context_pairs_for_training(
        data=data_for_validation, synsets_with_sense_ids=synsets,
        source_senses=source_senses, inflected_senses=inflected_senses,
        sense_occurrences=ruwordnet_occurrences, all_possible_pairs=False
    )
    file_name = os.path.join(destination_dir, 'contexts_for_validation.csv')
    save_context_pairs_to_csv(contexts_for_validation, file_name)
    print('{0} text pairs have been generated for validation.'.format(len(contexts_for_validation)))
    print('')
    del contexts_for_validation

    contexts_for_testing = generate_context_pairs_for_training(
        data=data_for_testing, synsets_with_sense_ids=synsets,
        source_senses=source_senses, inflected_senses=inflected_senses,
        sense_occurrences=ruwordnet_occurrences, all_possible_pairs=False
    )
    file_name = os.path.join(destination_dir, 'contexts_for_testing.csv')
    save_context_pairs_to_csv(contexts_for_testing, file_name)
    print('{0} text pairs have been generated for validation.'.format(len(contexts_for_testing)))
    print('')
    del contexts_for_testing


if __name__ == '__main__':
    main()
