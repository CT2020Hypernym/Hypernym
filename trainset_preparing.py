import random
from typing import Dict, List, Tuple

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
