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

import array
import multiprocessing
import os
import random
from typing import List, Sequence, Tuple, Union
import warnings

from bert.tokenization.bert_tokenization import FullTokenizer, validate_case_matches_checkpoint
from bert import BertModelLayer, params_from_pretrained_ckpt
from bert.loader import load_stock_weights
import numpy as np
import params_flow as pf
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.exceptions import UndefinedMetricWarning
import tensorflow as tf
import tensorflow_probability as tfp

from neural_network import MaskCalculator


MAX_SEQ_LENGTH = 512


def tokenize_text_pairs_for_bert(text_pairs: List[Tuple[str, str]], bert_tokenizer: FullTokenizer) -> \
        List[Tuple[Sequence[int], int]]:
    res = []
    for left_text, right_text in text_pairs:
        tokenized_left_text = bert_tokenizer.tokenize(left_text)
        if tokenized_left_text[0] != '[CLS]':
            tokenized_left_text = ['[CLS]'] + tokenized_left_text
        if tokenized_left_text[-1] != '[SEP]':
            tokenized_left_text = tokenized_left_text + ['[SEP]']
        tokenized_right_text = bert_tokenizer.tokenize(right_text)
        if tokenized_right_text[0] == '[CLS]':
            tokenized_right_text = tokenized_right_text[1:]
        if tokenized_right_text[-1] == '[SEP]':
            tokenized_right_text = tokenized_right_text[0:-1]
        tokenized_text = tokenized_left_text + tokenized_right_text
        if len(tokenized_text) > MAX_SEQ_LENGTH:
            warnings.warn(
                "The text pair `{0}` - `{1}` contains too many sub-tokens!".format(left_text, right_text))
            res.append((array.array("l"), 0))
        else:
            token_IDs = bert_tokenizer.convert_tokens_to_ids(tokenized_text)
            res.append((array.array("l", token_IDs), len(tokenized_left_text)))
        del tokenized_left_text, tokenized_right_text, tokenized_text
    return res


def tokenize_many_text_pairs_for_bert(text_pairs: List[Tuple[str, str, Union[str, int]]],
                                      bert_tokenizer: FullTokenizer, pool_: multiprocessing.Pool = None) -> \
        List[Tuple[Sequence[int], int, Union[str, int]]]:
    n_processes = os.cpu_count()
    res = []
    MAX_BUFFER_SIZE = 1000000
    if n_processes > 1:
        if pool_ is None:
            pool = multiprocessing.Pool(processes=n_processes)
        else:
            pool = pool_
        buffer_for_pairs = []
        buffer_for_additional_data = []
        for left_text, right_text, additional_data in text_pairs:
            buffer_for_pairs.append((left_text, right_text))
            buffer_for_additional_data.append(additional_data)
            if len(buffer_for_pairs) >= MAX_BUFFER_SIZE:
                n_data_part = int(np.ceil(len(buffer_for_pairs) / float(n_processes)))
                parts_of_buffer = [(buffer_for_pairs[(idx * n_data_part):((idx + 1) * n_data_part)], bert_tokenizer)
                                   for idx in range(n_processes - 1)]
                parts_of_buffer.append((buffer_for_pairs[((n_processes - 1) * n_data_part):], bert_tokenizer))
                pair_idx = 0
                for cur_part in pool.starmap(tokenize_text_pairs_for_bert, parts_of_buffer):
                    for token_IDs, n_left_tokens in cur_part:
                        if (len(token_IDs) > 0) and (n_left_tokens > 0):
                            if token_IDs[1:(n_left_tokens - 1)] != token_IDs[n_left_tokens:]:
                                res.append((token_IDs, n_left_tokens, buffer_for_additional_data[pair_idx]))
                        pair_idx += 1
                    del cur_part
                del parts_of_buffer
                buffer_for_pairs.clear()
                buffer_for_additional_data.clear()
            del left_text, right_text, additional_data
        if len(buffer_for_pairs) > 0:
            n_data_part = int(np.ceil(len(buffer_for_pairs) / float(n_processes)))
            parts_of_buffer = [(buffer_for_pairs[(idx * n_data_part):((idx + 1) * n_data_part)], bert_tokenizer)
                               for idx in range(n_processes - 1)]
            parts_of_buffer.append((buffer_for_pairs[((n_processes - 1) * n_data_part):], bert_tokenizer))
            pair_idx = 0
            for cur_part in pool.starmap(tokenize_text_pairs_for_bert, parts_of_buffer):
                for token_IDs, n_left_tokens in cur_part:
                    if (len(token_IDs) > 0) and (n_left_tokens > 0):
                        if token_IDs[1:(n_left_tokens - 1)] != token_IDs[n_left_tokens:]:
                            res.append((token_IDs, n_left_tokens, buffer_for_additional_data[pair_idx]))
                    pair_idx += 1
                del cur_part
            del parts_of_buffer
            buffer_for_pairs.clear()
            buffer_for_additional_data.clear()
        del buffer_for_pairs, buffer_for_additional_data
        if pool_ is None:
            del pool
    else:
        buffer_for_pairs = []
        buffer_for_additional_data = []
        for left_text, right_text, additional_data in text_pairs:
            buffer_for_pairs.append((left_text, right_text))
            buffer_for_additional_data.append(additional_data)
            if len(buffer_for_pairs) >= MAX_BUFFER_SIZE:
                for pair_idx, (token_IDs, n_left_tokens) in enumerate(tokenize_text_pairs_for_bert(buffer_for_pairs,
                                                                                                   bert_tokenizer)):
                    if (len(token_IDs) > 0) and (n_left_tokens > 0):
                        if token_IDs[1:(n_left_tokens - 1)] != token_IDs[n_left_tokens:]:
                            res.append((token_IDs, n_left_tokens, buffer_for_additional_data[pair_idx]))
                buffer_for_pairs.clear()
                buffer_for_additional_data.clear()
            del left_text, right_text, additional_data
        if len(buffer_for_pairs) > 0:
            for pair_idx, (token_IDs, n_left_tokens) in enumerate(tokenize_text_pairs_for_bert(buffer_for_pairs,
                                                                                               bert_tokenizer)):
                if (len(token_IDs) > 0) and (n_left_tokens > 0):
                    if token_IDs[1:(n_left_tokens - 1)] != token_IDs[n_left_tokens:]:
                        res.append((token_IDs, n_left_tokens, buffer_for_additional_data[pair_idx]))
        del buffer_for_pairs, buffer_for_additional_data
    return res


def calculate_optimal_number_of_tokens(lengths_of_texts: List[int]) -> int:
    lengths_of_texts.sort()
    print('A maximal number of sub-tokens in the BERT input is {0}.'.format(lengths_of_texts[-1]))
    print('A mean number of sub-tokens in the BERT input is {0}.'.format(
        int(round(sum(lengths_of_texts) / float(len(lengths_of_texts))))))
    print('A median number of sub-tokens in the BERT input is {0}.'.format(
        lengths_of_texts[len(lengths_of_texts) // 2]))
    n = int(round(0.85 * (len(lengths_of_texts) - 1)))
    print('85% of all texts are shorter then {0}.'.format(lengths_of_texts[n]))
    if n > 0:
        optimal_length = 4
        while optimal_length <= lengths_of_texts[n]:
            optimal_length *= 2
        optimal_length = min(optimal_length, MAX_SEQ_LENGTH)
        print('An optimal number of sub-tokens in the BERT input is {0}.'.format(optimal_length))
    else:
        optimal_length = MAX_SEQ_LENGTH
    return optimal_length


class TrainsetGenerator(tf.keras.utils.Sequence):
    def __init__(self, text_pairs: List[Tuple[Sequence[int], int, Union[int, str]]], seq_len: int, batch_size: int,
                 with_mask: bool = False):
        self.text_pairs = text_pairs
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.indices = list(range(len(self.text_pairs)))
        self.with_mask = with_mask
        random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.text_pairs) / float(self.batch_size)))

    def __getitem__(self, item):
        batch_start = item * self.batch_size
        indices_of_samples = [self.indices[idx]
                              for idx in range(batch_start, min(batch_start + self.batch_size, len(self.text_pairs)))]
        if len(indices_of_samples) < self.batch_size:
            indices_of_samples += random.sample(self.indices, self.batch_size - len(indices_of_samples))
        tokens = np.zeros((self.batch_size, self.seq_len), dtype=np.int32)
        segments = np.zeros((self.batch_size, self.seq_len), dtype=np.int32)
        if self.with_mask:
            mask = np.zeros((self.batch_size, self.seq_len,), dtype=np.int32)
        else:
            mask = None
        y = None
        for sample_idx, pair_idx in enumerate(indices_of_samples):
            token_ids, n_left_tokens, additional_data = self.text_pairs[pair_idx]
            for token_idx in range(len(token_ids)):
                tokens[sample_idx][token_idx] = token_ids[token_idx]
                segments[sample_idx][token_idx] = 1 if token_idx < n_left_tokens else 0
            if mask is not None:
                mask[sample_idx] = calculate_output_mask_for_bert(token_ids, n_left_tokens, self.seq_len)
            assert isinstance(additional_data, int) or isinstance(additional_data, str)
            if isinstance(additional_data, int):
                if y is None:
                    y = [additional_data]
                else:
                    y.append(additional_data)
            del token_ids, n_left_tokens, additional_data
        if y is None:
            del indices_of_samples
            if mask is None:
                return tokens, segments
            return tokens, segments, mask
        assert len(y) == len(indices_of_samples)
        del indices_of_samples
        if mask is None:
            return (tokens, segments), np.array(y, dtype=np.int32), [None]
        return (tokens, segments, mask), np.array(y, dtype=np.int32), [None]


def calculate_output_mask_for_bert(token_ids: Sequence[int], n_left_tokens: int, max_seq_len: int) -> np.ndarray:
    mask = np.zeros((max_seq_len,), dtype=np.float32)
    left_token_ids = token_ids[1:(n_left_tokens - 1)]
    right_token_ids = token_ids[n_left_tokens:]
    err_msg = '{0} is wrong sample!'.format((token_ids, n_left_tokens))
    assert (len(left_token_ids) > 0) and (len(right_token_ids) > 0), err_msg
    left_bounds = None
    right_bounds = None
    if len(left_token_ids) == len(right_token_ids):
        assert left_token_ids != right_token_ids, err_msg
    else:
        if len(left_token_ids) < len(right_token_ids):
            if left_token_ids == right_token_ids[:len(left_token_ids)]:
                left_bounds = (len(left_token_ids), len(left_token_ids) + 1)
                right_bounds = (len(left_token_ids) - 1 + n_left_tokens, len(right_token_ids) + n_left_tokens)
            elif left_token_ids == right_token_ids[(len(right_token_ids) - len(left_token_ids)):]:
                left_bounds = (1, 2)
                right_bounds = (n_left_tokens, len(right_token_ids) - len(left_token_ids) + 1 + n_left_tokens)
        else:
            if right_token_ids == left_token_ids[:len(right_token_ids)]:
                left_bounds = (len(right_token_ids), len(left_token_ids) + 1)
                right_bounds = (len(right_token_ids) - 1 + n_left_tokens, len(right_token_ids) + n_left_tokens)
            elif right_token_ids == left_token_ids[(len(left_token_ids) - len(right_token_ids)):]:
                right_bounds = (n_left_tokens, n_left_tokens + 1)
                left_bounds = (1, len(left_token_ids) - len(right_token_ids) + 2)
    assert ((left_bounds is None) and (right_bounds is None)) or \
           ((left_bounds is not None) and (right_bounds is not None)), err_msg
    if left_bounds is None:
        assert right_bounds is None, err_msg
        start_pos = -1
        for idx in range(min(len(left_token_ids), len(right_token_ids))):
            if left_token_ids[idx] != right_token_ids[idx]:
                start_pos = idx
                break
        assert start_pos >= 0, err_msg
        end_pos = -1
        for idx in range(min(len(left_token_ids), len(right_token_ids))):
            if left_token_ids[len(left_token_ids) - idx - 1] != right_token_ids[len(right_token_ids) - idx - 1]:
                end_pos = idx
                break
        assert end_pos >= 0, err_msg
        if start_pos > 0:
            start_pos -= 1
        if end_pos > 0:
            end_pos -= 1
        left_bounds = (start_pos + 1, n_left_tokens - end_pos)
        right_bounds = (n_left_tokens + start_pos, len(token_ids) - end_pos)
    else:
        assert right_bounds is not None, err_msg
    for idx in range(left_bounds[0], left_bounds[1]):
        mask[idx] = 1
    for idx in range(right_bounds[0], right_bounds[1]):
        mask[idx] = 1
    return mask


def create_dataset_for_bert(text_pairs: List[Tuple[Sequence[int], int, Union[int, str]]], seq_len: int,
                            batch_size: int, with_mask: bool = False) -> \
        Union[Tuple[np.ndarray, np.ndarray], Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray],
              Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]]:
    n_batches = int(np.ceil(len(text_pairs) / float(batch_size)))
    tokens = np.zeros((n_batches * batch_size, seq_len), dtype=np.int32)
    segments = np.zeros((n_batches * batch_size, seq_len), dtype=np.int32)
    if with_mask:
        mask = np.zeros((n_batches * batch_size, seq_len,), dtype=np.int32)
    else:
        mask = None
    indices = list(range(len(text_pairs)))
    if (n_batches * batch_size) > len(indices):
        indices += random.sample(indices, (n_batches * batch_size) - len(indices))
    y = None
    for sample_idx, pair_idx in enumerate(indices):
        token_ids, n_left_tokens, additional_data = text_pairs[pair_idx]
        for token_idx in range(len(token_ids)):
            tokens[sample_idx][token_idx] = token_ids[token_idx]
            segments[sample_idx][token_idx] = 1 if token_idx < n_left_tokens else 0
        if mask is not None:
            mask[sample_idx] = calculate_output_mask_for_bert(token_ids, n_left_tokens, seq_len)
        assert isinstance(additional_data, int) or isinstance(additional_data, str)
        if isinstance(additional_data, int):
            if y is None:
                y = [additional_data]
            else:
                y.append(additional_data)
        del token_ids, n_left_tokens, additional_data
    if y is None:
        del indices
        if mask is None:
            return tokens, segments
        return tokens, segments, mask
    assert len(y) == len(indices)
    del indices
    if mask is None:
        return (tokens, segments), np.array(y, dtype=np.int32)
    return (tokens, segments, mask), np.array(y, dtype=np.int32)


def initialize_tokenizer(model_dir: str) -> FullTokenizer:
    model_name = os.path.basename(model_dir)
    assert len(model_name) > 0, '`{0}` is wrong directory name for a BERT model.'.format(model_dir)
    bert_model_ckpt = os.path.join(model_dir, "bert_model.ckpt")
    do_lower_case = not ((model_name.lower().find("cased") == 0) or (model_name.lower().find("_cased") >= 0) or
                         (model_name.lower().find("-cased") >= 0))
    validate_case_matches_checkpoint(do_lower_case, bert_model_ckpt)
    vocab_file = os.path.join(model_dir, "vocab.txt")
    return FullTokenizer(vocab_file, do_lower_case)


def build_simple_bert(model_dir: str, max_seq_len: int, learning_rate: float, adapter_size: int = 64) -> tf.keras.Model:
    assert (max_seq_len > 0) and (max_seq_len <= MAX_SEQ_LENGTH)
    input_word_ids = tf.keras.layers.Input(shape=(max_seq_len,), dtype=tf.int32, name="input_word_ids_for_BERT")
    segment_ids = tf.keras.layers.Input(shape=(max_seq_len,), dtype=tf.int32, name="segment_ids_for_BERT")
    bert_params = params_from_pretrained_ckpt(model_dir)
    bert_params.adapter_size = adapter_size
    bert_params.adapter_init_scale = 1e-5
    bert_model_ckpt = os.path.join(model_dir, "bert_model.ckpt")
    bert_layer = BertModelLayer.from_params(bert_params, name="BERT_Layer")
    bert_output = bert_layer([input_word_ids, segment_ids])
    cls_output = tf.keras.layers.Lambda(lambda seq: seq[:, 0, :], name='BERT_cls')(bert_output)
    output_layer = tf.keras.layers.Dense(units=1, activation='sigmoid', kernel_initializer='glorot_uniform',
                                         name='HyponymHypernymOutput')(cls_output)
    model = tf.keras.Model(inputs=[input_word_ids, segment_ids], outputs=output_layer, name='SimpleBERT')
    model.build(input_shape=[(None, max_seq_len), (None, max_seq_len)])
    load_stock_weights(bert_layer, bert_model_ckpt)
    bert_layer.apply_adapter_freeze()
    bert_layer.embeddings_layer.trainable = False
    model.compile(optimizer=pf.optimizers.RAdam(learning_rate=learning_rate), loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.AUC(name='auc')], experimental_run_tf_function=True)
    return model


def build_bert_and_cnn(model_dir: str, n_filters: int, hidden_layer_size: int, bayesian: bool, max_seq_len: int,
                       learning_rate: float, **kwargs) -> tf.keras.Model:
    assert (max_seq_len > 0) and (max_seq_len <= MAX_SEQ_LENGTH)
    if bayesian:
        assert 'kl_weight' in kwargs
        print('KL weight is {0:.9f}.'.format(kwargs['kl_weight']))
    input_word_ids = tf.keras.layers.Input(shape=(max_seq_len,), dtype=tf.int32, name="input_word_ids_for_BERT")
    segment_ids = tf.keras.layers.Input(shape=(max_seq_len,), dtype=tf.int32, name="segment_ids_for_BERT")
    output_mask = tf.keras.layers.Input(shape=(max_seq_len,), dtype=tf.int32, name="output_mask_for_BERT")
    bert_params = params_from_pretrained_ckpt(model_dir)
    bert_model_ckpt = os.path.join(model_dir, "bert_model.ckpt")
    bert_layer = BertModelLayer.from_params(bert_params, name="BERT_Layer")
    bert_layer.trainable = False
    bert_output = bert_layer([input_word_ids, segment_ids])
    cls_output = tf.keras.layers.Lambda(lambda seq: seq[:, 0, :], name='BERT_cls')(bert_output)
    activation_type = 'tanh'
    initializer_type = 'glorot_uniform'
    if bayesian:
        kl_divergence_function = (
            lambda q, p, _: (tfp.distributions.kl_divergence(q, p) * tf.constant(kwargs['kl_weight'], dtype=tf.float32,
                                                                                 name='KL_weight'))
        )
        conv_layers = [
            tfp.layers.Convolution1DFlipout(kernel_size=1, filters=n_filters, activation=activation_type,
                                            kernel_divergence_fn=kl_divergence_function, padding='same',
                                            name='Conv_1grams')(bert_output),
            tfp.layers.Convolution1DFlipout(kernel_size=2, filters=n_filters, activation=activation_type,
                                            kernel_divergence_fn=kl_divergence_function, padding='same',
                                            name='Conv_2grams')(bert_output),
            tfp.layers.Convolution1DFlipout(kernel_size=3, filters=n_filters, activation=activation_type,
                                            kernel_divergence_fn=kl_divergence_function, padding='same',
                                            name='Conv_3grams')(bert_output),
            tfp.layers.Convolution1DFlipout(kernel_size=4, filters=n_filters, activation=activation_type,
                                            kernel_divergence_fn=kl_divergence_function, padding='same',
                                            name='Conv_4grams')(bert_output),
            tfp.layers.Convolution1DFlipout(kernel_size=5, filters=n_filters, activation=activation_type,
                                            kernel_divergence_fn=kl_divergence_function, padding='same',
                                            name='Conv_5grams')(bert_output),
        ]
    else:
        kl_divergence_function = None
        conv_layers = [
            tf.keras.layers.Conv1D(kernel_size=1, filters=n_filters, activation=activation_type, padding='same',
                                   kernel_initializer=initializer_type, name='Conv_1grams')(bert_output),
            tf.keras.layers.Conv1D(kernel_size=2, filters=n_filters, activation=activation_type, padding='same',
                                   kernel_initializer=initializer_type, name='Conv_2grams')(bert_output),
            tf.keras.layers.Conv1D(kernel_size=3, filters=n_filters, activation=activation_type, padding='same',
                                   kernel_initializer=initializer_type, name='Conv_3grams')(bert_output),
            tf.keras.layers.Conv1D(kernel_size=4, filters=n_filters, activation=activation_type, padding='same',
                                   kernel_initializer=initializer_type, name='Conv_4grams')(bert_output),
            tf.keras.layers.Conv1D(kernel_size=5, filters=n_filters, activation=activation_type, padding='same',
                                   kernel_initializer=initializer_type, name='Conv_5grams')(bert_output)
        ]
    conv_concat_layer = tf.keras.layers.Concatenate(name='ConvConcat')(conv_layers)
    masking_calc = MaskCalculator(output_dim=len(conv_layers) * n_filters, trainable=False,
                                  name='MaskCalculator')(output_mask)
    conv_concat_layer = tf.keras.layers.Multiply(name='MaskMultiplicator')([conv_concat_layer, masking_calc])
    conv_concat_layer = tf.keras.layers.Masking(name='Masking')(conv_concat_layer)
    feature_layer = tf.keras.layers.Concatenate(name='FeatureLayer')(
        [cls_output, tf.keras.layers.GlobalAveragePooling1D(name='AvePooling')(conv_concat_layer)]
    )
    if bayesian:
        hidden_layer = tfp.layers.DenseFlipout(units=hidden_layer_size, activation=activation_type,
                                               kernel_divergence_fn=kl_divergence_function,
                                               name='HiddenLayer')(feature_layer)
        output_layer = tfp.layers.DenseFlipout(units=1, activation='sigmoid',
                                               kernel_divergence_fn=kl_divergence_function,
                                               name='HyponymHypernymOutput')(hidden_layer)
    else:
        feature_layer = tf.keras.layers.Dropout(rate=0.5, name='Dropout1')(feature_layer)
        hidden_layer = tf.keras.layers.Dense(units=hidden_layer_size, activation=activation_type,
                                             kernel_initializer=initializer_type, name='HiddenLayer')(feature_layer)
        hidden_layer = tf.keras.layers.Dropout(rate=0.5, name='Dropout2')(hidden_layer)
        output_layer = tf.keras.layers.Dense(units=1, activation='sigmoid', kernel_initializer='glorot_uniform',
                                             name='HyponymHypernymOutput')(hidden_layer)
    model = tf.keras.Model(inputs=[input_word_ids, segment_ids, output_mask], outputs=output_layer, name='BERT_CNN')
    model.build(input_shape=[(None, max_seq_len), (None, max_seq_len), (None, max_seq_len)])
    load_stock_weights(bert_layer, bert_model_ckpt)
    model.compile(optimizer=pf.optimizers.RAdam(learning_rate=learning_rate), loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.AUC(name='auc')], experimental_run_tf_function=not bayesian)
    return model


def get_samples_per_epoch(trainset_generator: TrainsetGenerator, validset_generator: TrainsetGenerator) -> int:
    assert trainset_generator.batch_size == validset_generator.batch_size
    steps_per_epoch = min(len(trainset_generator), 3 * len(validset_generator))
    return steps_per_epoch * trainset_generator.batch_size


def train_neural_network(trainset_generator: TrainsetGenerator, validset_generator: TrainsetGenerator,
                         neural_network: tf.keras.Model, max_epochs: int, neural_network_name: str) -> tf.keras.Model:
    print('')
    print('Structure of neural network:')
    print('')
    neural_network.summary()
    print('')
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_auc', mode='max', restore_best_weights=True,
                                         verbose=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=neural_network_name, monitor='val_auc', mode='max',
                                           save_best_only=True, save_weights_only=True)
    ]
    steps_per_epoch = get_samples_per_epoch(trainset_generator, validset_generator) // trainset_generator.batch_size
    neural_network.fit(trainset_generator, validation_data=validset_generator, steps_per_epoch=steps_per_epoch,
                       epochs=max_epochs * max(1, len(trainset_generator.text_pairs) // steps_per_epoch), verbose=1,
                       callbacks=callbacks, shuffle=True)
    print('')
    return neural_network


def evaluate_neural_network(testset_generator: TrainsetGenerator, neural_network: tf.keras.Model,
                            num_monte_carlo: int = 0):
    assert isinstance(num_monte_carlo, int)
    assert num_monte_carlo >= 0
    if num_monte_carlo > 0:
        assert num_monte_carlo > 1
    y_true = []
    probabilities = []
    for batch_X, batch_y, _ in testset_generator:
        assert batch_y.shape[0] == batch_X[0].shape[0]
        y_true.append(batch_y)
        if num_monte_carlo > 0:
            new_probabilities = tf.reduce_mean(
                tf.stack([neural_network.predict_on_batch(batch_X) for _ in range(num_monte_carlo)]),
                axis=0
            )
        else:
            new_probabilities = neural_network.predict_on_batch(batch_X)
        probabilities.append(np.reshape(new_probabilities.numpy(), newshape=(batch_X[0].shape[0],)))
    y_true = np.concatenate(y_true)
    probabilities = np.concatenate(probabilities)
    print('Evaluation results:')
    print('  ROC-AUC is   {0:.6f}'.format(roc_auc_score(y_true, probabilities)))
    y_pred = np.asarray(probabilities >= 0.5, dtype=np.uint8)
    del probabilities
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        print('  Precision is {0:.6f}'.format(precision_score(y_true, y_pred, average='binary')))
        print('  Recall is    {0:.6f}'.format(recall_score(y_true, y_pred, average='binary')))
        print('  F1 is        {0:.6f}'.format(f1_score(y_true, y_pred, average='binary')))
    print('')
    del y_pred, y_true
