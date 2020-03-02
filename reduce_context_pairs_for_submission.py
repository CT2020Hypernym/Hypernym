from argparse import ArgumentParser
import array
import gzip
import os
import pickle
import random

import numpy as np


def main():
    random.seed(142)

    parser = ArgumentParser()
    parser.add_argument('-d', '--dst', dest='destination_dir', type=str, required=True,
                        help='A destination directory into which all results (as CSV files) will be saved.')
    parser.add_argument('-l', '--length', dest='sequence_length', type=int, required=True,
                        help='A maximal sequence length.')
    args = parser.parse_args()

    destination_dir = os.path.normpath(args.destination_dir)
    assert os.path.isdir(destination_dir), 'Directory `{0}` does not exist!'.format(destination_dir)
    assert os.path.isdir(os.path.join(destination_dir, 'context_pairs_for_public'))
    assert os.path.isdir(os.path.join(destination_dir, 'context_pairs_for_private'))
    assert args.sequence_length > 10
    assert args.sequence_length <= 512

    data = []
    names_of_files = list(map(
        lambda it2: os.path.join(destination_dir, 'context_pairs_for_public', it2),
        filter(lambda it1: it1.endswith(".gz"),
               os.listdir(os.path.join(destination_dir, 'context_pairs_for_public')))
    ))
    names_of_files.sort(key=lambda it: int(os.path.basename(it)[:-3]))
    n_data_parts = 8
    data_part_size = int(np.ceil(len(names_of_files) / float(n_data_parts)))
    start_pos = 0
    for file_name in names_of_files:
        print("Processing of `{0}` is started.".format(file_name))
        with gzip.open(file_name, "rb") as fp:
            contexts = pickle.load(fp)
        contexts_dict = dict()
        for tokens, n_left, synset_id in contexts:
            if len(tokens) <= args.sequence_length:
                contexts_dict[synset_id] = contexts_dict.get(synset_id, []) + [(array.array('l', tokens), n_left)]
        del contexts
        print('There are {0} synsets.'.format(len(contexts_dict)))
        contexts = []
        for synset_id in contexts_dict:
            values = contexts_dict[synset_id]
            values.sort(key=lambda it: (len(it[0]), it[1]))
            if len(values) > 2:
                values = [values[0], values[-1]]
            for tokens, n_left in values:
                contexts.append((tokens, n_left, synset_id))
        data.append(tuple(contexts))
        del contexts, contexts_dict
        print("Processing of `{0}` is finished.".format(file_name))
        print('')
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
            'context_pairs_for_public_{0}_{1}.pkl'.format(start_pos, len(data) + start_pos)
        )
        with open(data_file_name, 'wb') as fp:
            pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
        start_pos += len(data)
        data.clear()
    del data

    data = []
    names_of_files = list(map(
        lambda it2: os.path.join(destination_dir, 'context_pairs_for_private', it2),
        filter(lambda it1: it1.endswith(".gz"),
               os.listdir(os.path.join(destination_dir, 'context_pairs_for_private')))
    ))
    names_of_files.sort(key=lambda it: int(os.path.basename(it)[:-3]))
    n_data_parts = 8
    data_part_size = int(np.ceil(len(names_of_files) / float(n_data_parts)))
    start_pos = 0
    for file_name in names_of_files:
        print("Processing of `{0}` is started.".format(file_name))
        with gzip.open(file_name, "rb") as fp:
            contexts = pickle.load(fp)
        contexts_dict = dict()
        for tokens, n_left, synset_id in contexts:
            if len(tokens) <= args.sequence_length:
                contexts_dict[synset_id] = contexts_dict.get(synset_id, []) + [(array.array('l', tokens), n_left)]
        del contexts
        print('There are {0} synsets.'.format(len(contexts_dict)))
        contexts = []
        for synset_id in contexts_dict:
            values = contexts_dict[synset_id]
            values.sort(key=lambda it: (len(it[0]), it[1]))
            if len(values) > 2:
                values = [values[0], values[-1]]
            for tokens, n_left in values:
                contexts.append((tokens, n_left, synset_id))
        data.append(tuple(contexts))
        del contexts, contexts_dict
        print("Processing of `{0}` is finished.".format(file_name))
        print('')
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
    del data


if __name__ == '__main__':
    main()
