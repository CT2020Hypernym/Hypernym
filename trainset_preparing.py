import codecs
import csv
import random
from typing import Dict, List, Tuple, Union

from gensim.test.utils import datapath
from gensim.models.fasttext import load_facebook_model
import numpy as np
import tensorflow as tf

from ruwordnet_parsing import TrainingData


def calculate_word_embeddings(all_words: Dict[str, int], fasttext_model_path: str) -> np.ndarray:
    EPS = 1e-2
    indices = sorted(list(set(map(lambda token: all_words[token], all_words.keys()))))
    assert len(indices) == len(all_words)
    assert indices[0] == 1
    assert indices[-1] == len(indices)
    del indices
    fasttext_model = load_facebook_model(datapath(fasttext_model_path))
    embedding_size = fasttext_model['1'].shape[0]
    embeddings_matrix = np.zeros((len(all_words) + 1, embedding_size), dtype=np.float32)
    n_nonzeros = 0
    for token in all_words:
        token_idx = all_words[token]
        try:
            token_vector = np.copy(fasttext_model[token])
        except:
            token_vector = None
        if token_vector is not None:
            vector_norm = np.linalg.norm(token_vector)
            assert vector_norm > 0.0
            token_vector /= vector_norm
            vector_norm = np.linalg.norm(token_vector)
            assert (vector_norm > (1.0 - EPS)) and (vector_norm < (1.0 + EPS))
            embeddings_matrix[token_idx] = token_vector
            n_nonzeros += 1
    del fasttext_model
    print('{0} word vectors have been calculated. The number of non-zero word vectors is {1}.'.format(
        embeddings_matrix.shape[0] - 1, n_nonzeros))
    return embeddings_matrix


def get_maximal_lengths_of_texts(synsets: Dict[str, Tuple[List[tuple], tuple]]) -> Tuple[int, int]:
    max_hyponym_length = 0
    max_hypernym_length = 0
    for synset_id in synsets:
        synonyms, description = synsets[synset_id]
        max_hyponym_length_ = max(map(lambda it: len(it), synonyms))
        if max_hyponym_length_ > max_hyponym_length:
            max_hyponym_length = max_hyponym_length_
        if len(description) > 0:
            if len(description) > max_hypernym_length:
                max_hypernym_length = len(description)
            else:
                max_hypernym_length_ = len(synonyms[0])
                for cur in synonyms[1:]:
                    max_hypernym_length_ += (len(cur) + 1)
                if max_hypernym_length_ > max_hypernym_length:
                    max_hypernym_length = max_hypernym_length_
    return max_hyponym_length, max_hypernym_length


def hyponym_to_text(synset_id: str, synsets: Dict[str, Tuple[List[tuple], tuple]], deterministic: bool = True) -> tuple:
    if deterministic:
        hyponym_tokens = synsets[synset_id][0][0]
    else:
        hyponym_tokens = random.choice(synsets[synset_id][0])
    return hyponym_tokens


def hypernym_to_text(synset_id: str, synsets: Dict[str, Tuple[List[tuple], tuple]]) -> tuple:
    if len(synsets[synset_id][1]) > 0:
        hypernym_tokens = synsets[synset_id][1]
    else:
        hypernym_tokens = list(synsets[synset_id][0][0])
        for synonym in synsets[synset_id][0][1:]:
            hypernym_tokens += [';'] + list(synonym)
    return hypernym_tokens


class TrainsetGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_for_training: TrainingData, synsets: Dict[str, Tuple[List[tuple], tuple]],
                 tokens_dict: Dict[str, int], max_hyponym_length: int, max_hypernym_length: int, batch_size: int,
                 deterministic: bool):
        assert len(data_for_training.hyponyms) == len(data_for_training.hypernyms)
        assert len(data_for_training.hyponyms) == len(data_for_training.is_true)
        assert len(data_for_training.is_true) > 0
        self.data_for_training = data_for_training
        self.synsets = synsets
        self.tokens_dict = tokens_dict
        self.max_hyponym_length = max_hyponym_length
        self.max_hypernym_length = max_hypernym_length
        self.batch_size = batch_size
        self.deterministic = deterministic
        self.n_batches = int(np.ceil(len(self.data_for_training.is_true) / float(self.batch_size)))
        self.batch_indices = np.arange(0, self.n_batches, 1, dtype=np.int32)
        np.random.shuffle(self.batch_indices)
        self.batch_counter = 0

    def __len__(self):
        return self.n_batches

    def __getitem__(self, item):
        if self.deterministic:
            batch_start = item * self.batch_size
        else:
            batch_start = self.batch_indices[item] * self.batch_size
        batch_end = min(batch_start + self.batch_size, len(self.data_for_training.is_true))
        batch_size = batch_end - batch_start
        X1 = np.zeros((batch_size, self.max_hyponym_length), dtype=np.int32)
        X2 = np.zeros((batch_size, self.max_hypernym_length), dtype=np.int32)
        y = np.zeros((batch_size,), dtype=np.int32)
        for sample_idx in range(batch_start, batch_end):
            hyponym_tokens = hyponym_to_text(self.data_for_training.hyponyms[sample_idx], self.synsets,
                                             deterministic=self.deterministic)
            hypernym_tokens = hypernym_to_text(self.data_for_training.hypernyms[sample_idx], self.synsets)
            n = min(len(hyponym_tokens), self.max_hyponym_length)
            for token_idx, token_text in enumerate(hyponym_tokens[0:n]):
                X1[sample_idx - batch_start][token_idx] = self.tokens_dict[token_text]
            n = min(len(hypernym_tokens), self.max_hypernym_length)
            for token_idx, token_text in enumerate(hypernym_tokens[0:n]):
                X2[sample_idx - batch_start][token_idx] = self.tokens_dict[token_text]
            y[sample_idx - batch_start] = self.data_for_training.is_true[sample_idx]
        self.batch_counter += 1
        if self.batch_counter >= self.n_batches:
            self.batch_counter = 0
            np.random.shuffle(self.batch_indices)
        return (X1, X2), y


def build_dataset_for_training(data: TrainingData, synsets: Dict[str, Tuple[List[tuple], tuple]],
                               tokens_dict: Dict[str, int], max_hyponym_length: int, max_hypernym_length: int) -> \
        Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    assert len(data.hyponyms) == len(data.hypernyms)
    assert len(data.hyponyms) == len(data.is_true)
    assert len(data.is_true) > 0
    X1 = np.zeros((len(data.is_true), max_hyponym_length), dtype=np.int32)
    X2 = np.zeros((len(data.is_true), max_hypernym_length), dtype=np.int32)
    y = np.zeros((len(data.is_true),), dtype=np.int32)
    for sample_idx in range(y.shape[0]):
        hyponym_tokens = hyponym_to_text(data.hyponyms[sample_idx], synsets, deterministic=True)
        hypernym_tokens = hypernym_to_text(data.hypernyms[sample_idx], synsets)
        n = min(len(hyponym_tokens), max_hyponym_length)
        for token_idx, token_text in enumerate(hyponym_tokens[0:n]):
            X1[sample_idx][token_idx] = tokens_dict[token_text]
        n = min(len(hypernym_tokens), max_hypernym_length)
        for token_idx, token_text in enumerate(hypernym_tokens[0:n]):
            X2[sample_idx][token_idx] = tokens_dict[token_text]
        y[sample_idx] = data.is_true[sample_idx]
    return (X1, X2), y


def build_dataset_for_submission(unseen_hyponym: tuple, synsets: Dict[str, Tuple[List[tuple], tuple]],
                                 tokens_dict: Dict[str, int], max_hyponym_length: int, max_hypernym_length: int) -> \
        Tuple[Tuple[np.ndarray, np.ndarray], List[str]]:
    synset_IDs = sorted(list(synsets.keys()))
    n = min(len(unseen_hyponym), max_hyponym_length)
    X1 = np.zeros((1, max_hyponym_length), dtype=np.int32)
    for token_idx, token_text in enumerate(unseen_hyponym[0:n]):
        X1[0][token_idx] = tokens_dict[token_text]
    X2 = np.zeros((len(synset_IDs), max_hypernym_length), dtype=np.int32)
    for sample_idx in range(len(synset_IDs)):
        hypernym_tokens = hypernym_to_text(synset_IDs[sample_idx], synsets)
        n = min(len(hypernym_tokens), max_hypernym_length)
        for token_idx, token_text in enumerate(hypernym_tokens[0:n]):
            X2[sample_idx][token_idx] = tokens_dict[token_text]
    return (np.repeat(X1, repeats=len(synset_IDs), axis=0), X2), synset_IDs


def generate_context_pairs_for_training(data: TrainingData, synsets_with_sense_ids: Dict[str, List[str]],
                                        source_senses: Dict[str, str],
                                        inflected_senses: Dict[str, Dict[str, Tuple[tuple, Tuple[int, int]]]],
                                        sense_occurrences: Dict[str, Dict[str, List[Tuple[str, Tuple[int, int]]]]],
                                        all_possible_pairs: bool) -> \
        List[Tuple[str, str, int]]:
    """ Generate all possible training pairs of texts for existing hyponym-hypernym relations.

    For all pairs "hyponym - candidate for hypernym" described by the `data` argument, we find occurrences of
    the corresponded hyponym in real-world texts, after that we prepare analogous occurrence of
    the candidate for hypernym by replacing source hyponym with this candidate for hypernym.

    :param data: a Python tuple, which consists of three lists: hyponym IDs, hypernym IDs, and binary relation labels.
    :param synsets_with_sense_ids: a Python dictionary, which describes sense IDs for each synset ID.
    :param source_senses: a Python dictionary of normalized sense texts by their IDs.
    :param inflected_senses: inflected senses dictionary, returned by the `ruwordnet_parsing.load_and_inflect_senses`.
    :param sense_occurrences: occurrences of most of the inflected senses in real-world texts (from Wiki, news, etc.).
    :param all_possible_pairs: the flag which shows need to generate all possible combinations for synset pairs.
    :return: list of 3-element tuples: text with hyponym, text with candidate for a hypernym, and binary relation label.
    """
    assert len(data.hyponyms) == len(data.hypernyms)
    assert len(data.hyponyms) == len(data.is_true)
    assert len(data.is_true) > 0
    text_pairs_and_labels = []
    for sample_idx in range(len(data.is_true)):
        hyponym_synset_ID = data.hyponyms[sample_idx]
        hypernym_synset_ID = data.hypernyms[sample_idx]
        y = data.is_true[sample_idx]
        for hyponym_sense_ID in synsets_with_sense_ids[hyponym_synset_ID]:
            for hypernym_sense_ID in synsets_with_sense_ids[hypernym_synset_ID]:
                text_pairs_and_labels.append((source_senses[hyponym_sense_ID], source_senses[hypernym_sense_ID], y))
                new_pairs = []
                if (hyponym_sense_ID in sense_occurrences) and (hypernym_sense_ID in inflected_senses):
                    for morphotag in sense_occurrences[hyponym_sense_ID]:
                        if morphotag in inflected_senses[hypernym_sense_ID]:
                            hypernym = list(inflected_senses[hypernym_sense_ID][morphotag][0])
                            for text_with_hyponym, hyponym_bounds in sense_occurrences[hyponym_sense_ID][morphotag]:
                                text_with_hypernym = text_with_hyponym.split()
                                text_with_hypernym = text_with_hypernym[0:hyponym_bounds[0]] + hypernym + \
                                                     text_with_hypernym[hyponym_bounds[1]:]
                                new_pairs.append((text_with_hyponym, ' '.join(text_with_hypernym), y))
                if len(new_pairs) > 0:
                    if all_possible_pairs:
                        if len(new_pairs) > 10:
                            random.shuffle(new_pairs)
                            text_pairs_and_labels += new_pairs[0:10]
                        else:
                            text_pairs_and_labels += new_pairs
                    else:
                        if len(new_pairs) > 2:
                            new_pairs.sort(key=lambda it: max(len(it[0]), len(it[1])))
                            text_pairs_and_labels.append(new_pairs[0])
                            text_pairs_and_labels.append(random.choice(new_pairs[1:]))
                        else:
                            text_pairs_and_labels += new_pairs
                del new_pairs
    random.shuffle(text_pairs_and_labels)
    return text_pairs_and_labels


def save_context_pairs_to_csv(pairs: List[Tuple[str, str, int]], file_name: str):
    """ Save text pairs, generated by the `generate_context_pairs_for_training` function, into a CSV file.

    :param pairs: List of 3-element tuples, each of them contains text pair and relation label.
    :param file_name: Name of the CSV file.
    :return:
    """
    header = ["Context of hyponym", "Context of candidate for hypernym", "Is true hypernym?"]
    with codecs.open(file_name, mode='w', encoding='utf-8', errors='ignore') as fp:
        data_writer = csv.writer(fp, quotechar='"', delimiter=',')
        data_writer.writerow(header)
        for left_text, right_text, label in pairs:
            data_writer.writerow([left_text, right_text, str(label)])


def load_context_pairs_from_csv(file_name: str) -> List[Tuple[str, str, int]]:
    """ Load text pairs, generated by the `generate_context_pairs_for_training` function, from a CSV file.

    :param file_name: Name of the CSV file.
    :return: List of 3-element tuples, each of them contains text pair and relation label.
    """
    true_header = ["Context of hyponym", "Context of candidate for hypernym", "Is true hypernym?"]
    loaded_header = []
    line_idx = 1
    with codecs.open(file_name, mode='r', encoding='utf-8', errors='ignore') as fp:
        data_reader = csv.reader(fp, quotechar='"', delimiter=',')
        for row in data_reader:
            if len(row) > 0:
                err_msg = 'File `{0}`: line {1} is wrong!'.format(file_name, line_idx)
                if len(loaded_header) == 0:
                    loaded_header = row
                    assert loaded_header == true_header, err_msg + ' {0} != {1}'.format(true_header, loaded_header)
                else:
                    assert len(row) == len(true_header), err_msg
                    try:
                        label = int(row[2])
                        if label not in {0, 1}:
                            label = None
                    except:
                        label = None
                    assert label is not None, err_msg
                    left_text = row[0].strip()
                    right_text = row[1].strip()
                    assert (len(left_text) > 0) and (len(right_text) > 0), err_msg
                    yield left_text, right_text, label
            line_idx += 1


def generate_context_pairs_for_submission(unseen_hyponym: tuple,
                                          occurrences_of_hyponym: Dict[str, List[Tuple[str, Tuple[int, int]]]],
                                          synsets_with_sense_ids: Dict[str, List[str]],
                                          source_senses: Dict[str, str],
                                          inflected_senses: Dict[str, Dict[str, Tuple[tuple, Tuple[int, int]]]],
                                          checked_synsets: Union[List[str], None] = None) -> \
        List[Tuple[str, str, str]]:
    """ Generate all possible pairs of texts for the specified unseen hyponym and all hypernyms from the RuWordNet.

    This function is similar to the `generate_context_pairs_for_training`, but we don't know about
    hyponym-hypernym relation labels for text pairs. As in the aforementioned function, we also form a list,
    where each item is a 3-element tuple. The first and second elements of this tuple are the same as
    analogous elements of the tuple in the `generate_context_pairs_for_training` function, but the third element is
    a hypernym ID from RuWordNet instead of a binary relation label.

    :param unseen_hyponym: tokenized text of unseen hyponym from the public or the private submission set.
    :param occurrences_of_hyponym: all occurrences of this hyponym in real-word texts.
    :param synsets_with_sense_ids: a Python dictionary, which describes sense IDs for each synset ID.
    :param source_senses: a Python dictionary of normalized sense texts by their IDs.
    :param inflected_senses: inflected senses dictionary, returned by the `ruwordnet_parsing.load_and_inflect_senses`.
    :param checked_synsets: a list of synset IDs for checking (if it is specified, then we don't check all RuWordNet).
    :return: list of 3-element tuples: text with hyponym, text with hypernym candidate, and synset ID of this candidate.
    """
    text_pairs = []
    all_synset_IDs = sorted(list(synsets_with_sense_ids.keys())) if checked_synsets is None else checked_synsets
    if (occurrences_of_hyponym is not None) and (len(occurrences_of_hyponym) > 0):
        for hypernym_synset_ID in all_synset_IDs:
            new_pairs = []
            for hypernym_sense_ID in synsets_with_sense_ids[hypernym_synset_ID]:
                if hypernym_sense_ID in inflected_senses:
                    for morphotag in occurrences_of_hyponym:
                        if morphotag in inflected_senses[hypernym_sense_ID]:
                            hypernym = list(inflected_senses[hypernym_sense_ID][morphotag][0])
                            for text_with_hyponym, hyponym_bounds in occurrences_of_hyponym[morphotag]:
                                text_with_hypernym = text_with_hyponym.split()
                                text_with_hypernym = text_with_hypernym[0:hyponym_bounds[0]] + hypernym + \
                                                     text_with_hypernym[hyponym_bounds[1]:]
                                text_with_hypernym = ' '.join(text_with_hypernym)
                                new_pairs.append((text_with_hyponym, text_with_hypernym, hypernym_synset_ID))
            if len(new_pairs) > 0:
                if len(new_pairs) > 3:
                    new_pairs.sort(key=lambda it: max(len(it[0]), len(it[1])))
                    text_pairs.append(new_pairs[0])
                    text_pairs += random.sample(new_pairs[1:], 2)
                else:
                    text_pairs += new_pairs
            del new_pairs
    for hypernym_synset_ID in all_synset_IDs:
        new_pairs = []
        for hypernym_sense_ID in synsets_with_sense_ids[hypernym_synset_ID]:
            new_pairs.append((' '.join(unseen_hyponym), source_senses[hypernym_sense_ID], hypernym_synset_ID))
        if len(new_pairs) > 0:
            if len(new_pairs) > 2:
                random.shuffle(new_pairs)
                text_pairs += new_pairs[:2]
            else:
                text_pairs += new_pairs
        del new_pairs
    return text_pairs
