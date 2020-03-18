"""
This module is a part of system for the automatic enrichment
of a WordNet-like taxonomy.

Copyright 2020 Ivan Bondarenko

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
import os
import random
from typing import List

from gensim.test.utils import datapath
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.fasttext import load_facebook_model
import numpy as np

from text_processing import load_sense_occurrences_in_texts


def load_text_corpus(file_name: str) -> List[List[str]]:
    texts = []
    occurrences = load_sense_occurrences_in_texts(file_name)
    for sense_id in occurrences:
        for morpho_tag in occurrences[sense_id]:
            texts += [it[0].split() for it in occurrences[sense_id][morpho_tag]]
    return texts


class EpochLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch_counter = 0

    def on_epoch_end(self, model):
        self.epoch_counter += 1
        print('{0} epochs of training have been executed...'.format(self.epoch_counter))


def main():
    random.seed(142)
    np.random.seed(142)

    parser = ArgumentParser()
    parser.add_argument('-s', '--src', dest='source_fasttext_name', type=str, required=True,
                        help='A binary file with a source Facebook-like FastText model (*.bin).')
    parser.add_argument('-d', '--dst', dest='destination_fasttext_name', type=str, required=True,
                        help='A binary file with a destination Facebook-like FastText model (*.bin) after '
                             'its fine-tuning.')
    parser.add_argument('-c', '--cache_dir', dest='cache_dir', type=str, required=True,
                        help='A directory with cached data for training.')
    parser.add_argument('--epochs', dest='max_epochs', type=int, required=False, default=5,
                        help='A number of epochs to train the FastText model.')
    args = parser.parse_args()

    source_fasttext_name = os.path.normpath(args.source_fasttext_name)
    assert os.path.isfile(source_fasttext_name), 'File `{0}` does not exist!'.format(source_fasttext_name)
    destination_fasttext_name = os.path.normpath(args.destination_fasttext_name)
    destination_fasttext_dir = os.path.dirname(destination_fasttext_name)
    assert os.path.isdir(destination_fasttext_dir), 'Directory `{0}` does not exist!'.format(destination_fasttext_dir)
    cache_data_dir = os.path.normpath(args.cache_dir)
    assert os.path.isdir(cache_data_dir), 'Directory `{0}` does not exist!'.format(cache_data_dir)
    ruwordnet_occurrences_file = os.path.join(cache_data_dir, 'ruwordnet_occurrences_in_texts.json')
    submission_occurrences_file = os.path.join(cache_data_dir, 'submission_occurrences_in_texts.json')
    assert os.path.isfile(ruwordnet_occurrences_file), 'File `{0}` does not exist!'.format(ruwordnet_occurrences_file)
    assert os.path.isfile(submission_occurrences_file), 'File `{0}` does not exist!'.format(submission_occurrences_file)

    texts = load_text_corpus(ruwordnet_occurrences_file) + load_text_corpus(submission_occurrences_file)
    print('{0} texts have been loaded...'.format(len(texts)))
    fasttext_model = load_facebook_model(datapath(source_fasttext_name))
    print('The FastText model has been loaded...')
    fasttext_model.workers = max(os.cpu_count(), 1)
    fasttext_model.min_count = 1
    fasttext_model.callbacks = [EpochLogger()] + list(fasttext_model.callbacks)
    fasttext_model.build_vocab(texts, update=True)
    fasttext_model.train(texts, total_examples=len(texts), epochs=args.max_epochs)
    fasttext_model.callbacks = ()
    fasttext_model.save(destination_fasttext_name)


if __name__ == '__main__':
    main()
