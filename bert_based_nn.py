import array
import codecs
import csv
import multiprocessing
import os
import random
from typing import Dict, List, Sequence, Tuple, Union
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
from trainset_preparing import generate_context_pairs_for_submission


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
            res.append((array.array('l'), 0))
        else:
            token_IDs = bert_tokenizer.convert_tokens_to_ids(tokenized_text)
            res.append((array.array('l', token_IDs), len(tokenized_left_text)))
    return res


def tokenize_many_text_pairs_for_bert(text_pairs: List[Tuple[str, str, Union[str, int]]],
                                      bert_tokenizer: FullTokenizer) -> \
        List[Tuple[Sequence[int], int, Union[str, int]]]:
    n_processes = os.cpu_count()
    res = []
    MAX_BUFFER_SIZE = 1000000
    if n_processes > 1:
        pool = multiprocessing.Pool(processes=n_processes)
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
                parts_of_result = list(pool.starmap(tokenize_text_pairs_for_bert, parts_of_buffer))
                del parts_of_buffer
                pair_idx = 0
                for cur_part in parts_of_result:
                    for token_IDs, n_left_tokens in cur_part:
                        if (len(token_IDs) > 0) and (n_left_tokens > 0):
                            res.append((token_IDs, n_left_tokens, buffer_for_additional_data[pair_idx]))
                        pair_idx += 1
                del parts_of_result
                buffer_for_pairs.clear()
                buffer_for_additional_data.clear()
            del left_text, right_text, additional_data
        if len(buffer_for_pairs) > 0:
            n_data_part = int(np.ceil(len(buffer_for_pairs) / float(n_processes)))
            parts_of_buffer = [(buffer_for_pairs[(idx * n_data_part):((idx + 1) * n_data_part)], bert_tokenizer)
                               for idx in range(n_processes - 1)]
            parts_of_buffer.append((buffer_for_pairs[((n_processes - 1) * n_data_part):], bert_tokenizer))
            parts_of_result = list(pool.starmap(tokenize_text_pairs_for_bert, parts_of_buffer))
            del parts_of_buffer
            pair_idx = 0
            for cur_part in parts_of_result:
                for token_IDs, n_left_tokens in cur_part:
                    if (len(token_IDs) > 0) and (n_left_tokens > 0):
                        res.append((token_IDs, n_left_tokens, buffer_for_additional_data[pair_idx]))
                    pair_idx += 1
            del parts_of_result
            buffer_for_pairs.clear()
            buffer_for_additional_data.clear()
        del buffer_for_pairs, buffer_for_additional_data
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
                        res.append((token_IDs, n_left_tokens, buffer_for_additional_data[pair_idx]))
                buffer_for_pairs.clear()
                buffer_for_additional_data.clear()
            del left_text, right_text, additional_data
        if len(buffer_for_pairs) > 0:
            for pair_idx, (token_IDs, n_left_tokens) in enumerate(tokenize_text_pairs_for_bert(buffer_for_pairs,
                                                                                               bert_tokenizer)):
                if (len(token_IDs) > 0) and (n_left_tokens > 0):
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
    n = int(round(0.75 * (len(lengths_of_texts) - 1)))
    print('75% of all texts are shorter then {0}.'.format(lengths_of_texts[n]))
    if n > 0:
        optimal_length = 4
        while optimal_length <= lengths_of_texts[n]:
            optimal_length *= 2
        optimal_length //= 2
        optimal_length = min(optimal_length, MAX_SEQ_LENGTH)
        print('An optimal number of sub-tokens in the BERT input is {0}.'.format(optimal_length))
    else:
        optimal_length = MAX_SEQ_LENGTH
    return optimal_length


class BertDatasetGenerator(tf.keras.utils.Sequence):
    def __init__(self, text_pairs: List[Tuple[Sequence[int], int, Union[int, str]]], batch_size: int, seq_len: int,
                 return_y: bool = True):
        self.text_pairs = text_pairs
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.return_y = return_y

    def __len__(self):
        return int(np.ceil(len(self.text_pairs) / float(self.batch_size)))

    def __getitem__(self, item):
        batch_start = item * self.batch_size
        batch_end = min(batch_start + self.batch_size, len(self.text_pairs))
        tokens = np.zeros((batch_end - batch_start, self.seq_len), dtype=np.int32)
        segments = np.zeros((batch_end - batch_start, self.seq_len), dtype=np.int32)
        y = None
        for sample_idx in range(batch_start, batch_end):
            token_ids, n_left_tokens, additional_data = self.text_pairs[sample_idx]
            for token_idx in range(len(token_ids)):
                tokens[sample_idx - batch_start][token_idx] = token_ids[token_idx]
                segments[sample_idx - batch_start][token_idx] = 1 if token_idx < n_left_tokens else 0
            assert isinstance(additional_data, int) or isinstance(additional_data, str)
            if isinstance(additional_data, int):
                if y is None:
                    y = [additional_data]
                else:
                    y.append(additional_data)
        if y is None:
            return tokens, segments
        assert len(y) == (batch_end - batch_start)
        if not self.return_y:
            return tokens, segments
        return (tokens, segments), np.array(y, dtype=np.int32), [None]


def create_dataset_for_bert(text_pairs: List[Tuple[Sequence[int], int, Union[int, str]]], seq_len: int,
                            batch_size: int) -> \
        Union[Tuple[np.ndarray, np.ndarray], Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]]:
    n_batches = int(np.ceil(len(text_pairs) / float(batch_size)))
    tokens = np.zeros((n_batches * batch_size, seq_len), dtype=np.int32)
    segments = np.zeros((n_batches * batch_size, seq_len), dtype=np.int32)
    indices = list(range(len(text_pairs)))
    if (n_batches * batch_size) > len(indices):
        indices += random.sample(indices, (n_batches * batch_size) - len(indices))
    y = None
    for sample_idx, pair_idx in enumerate(indices):
        token_ids, n_left_tokens, additional_data = text_pairs[pair_idx]
        for token_idx in range(len(token_ids)):
            tokens[sample_idx][token_idx] = token_ids[token_idx]
            segments[sample_idx][token_idx] = 1 if token_idx < n_left_tokens else 0
        assert isinstance(additional_data, int) or isinstance(additional_data, str)
        if isinstance(additional_data, int):
            if y is None:
                y = [additional_data]
            else:
                y.append(additional_data)
    if y is None:
        return tokens, segments
    assert len(y) == len(indices)
    return (tokens, segments), np.array(y, dtype=np.int32)


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


def build_bert_and_cnn(model_dir: str, n_filters: int, hidden_layer_size: int, ave_pooling: bool, bayesian: bool,
                       max_seq_len: int, learning_rate: float, **kwargs) -> tf.keras.Model:
    assert (max_seq_len > 0) and (max_seq_len <= MAX_SEQ_LENGTH)
    if bayesian:
        assert 'kl_weight' in kwargs
    input_word_ids = tf.keras.layers.Input(shape=(max_seq_len,), dtype=tf.int32, name="input_word_ids_for_BERT")
    segment_ids = tf.keras.layers.Input(shape=(max_seq_len,), dtype=tf.int32, name="segment_ids_for_BERT")
    bert_params = params_from_pretrained_ckpt(model_dir)
    bert_model_ckpt = os.path.join(model_dir, "bert_model.ckpt")
    bert_layer = BertModelLayer.from_params(bert_params, name="BERT_Layer")
    bert_layer.trainable = False
    bert_output = bert_layer([input_word_ids, segment_ids])
    cls_output = tf.keras.layers.Lambda(lambda seq: seq[:, 0, :], name='BERT_cls')(bert_output)
    activation_type = 'tanh' if ave_pooling else 'elu'
    initializer_type = 'glorot_uniform' if ave_pooling else 'he_uniform'
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
    if ave_pooling:
        masking_calc = MaskCalculator(output_dim=len(conv_layers) * n_filters, trainable=False,
                                      name='MaskCalculator')(input_word_ids)
        conv_concat_layer = tf.keras.layers.Multiply(name='MaskMultiplicator')([conv_concat_layer, masking_calc])
        conv_concat_layer = tf.keras.layers.Masking(name='Masking')(conv_concat_layer)
        feature_layer = tf.keras.layers.Concatenate(name='FeatureLayer')(
            [cls_output, tf.keras.layers.GlobalAveragePooling1D(name='AvePooling')(conv_concat_layer)]
        )
    else:
        feature_layer = tf.keras.layers.Concatenate(name='FeatureLayer')(
            [cls_output, tf.keras.layers.GlobalMaxPooling1D(name='MaxPooling')(conv_concat_layer)]
        )
    if bayesian:
        hidden_layer = tfp.layers.DenseFlipout(units=hidden_layer_size, activation=activation_type,
                                               kernel_divergence_fn=kl_divergence_function,
                                               name='HiddenLayer')(feature_layer)
        output_layer = tf.keras.layers.Dense(units=1, activation='sigmoid', kernel_divergence_fn=kl_divergence_function,
                                             name='HyponymHypernymOutput')(hidden_layer)
    else:
        feature_layer = tf.keras.layers.Dropout(rate=0.5, name='Dropout1')(feature_layer)
        hidden_layer = tf.keras.layers.Dense(units=hidden_layer_size, activation=activation_type,
                                             kernel_initializer=initializer_type, name='HiddenLayer')(feature_layer)
        hidden_layer = tf.keras.layers.Dropout(rate=0.5, name='Dropout2')(hidden_layer)
        output_layer = tf.keras.layers.Dense(units=1, activation='sigmoid', kernel_initializer='glorot_uniform',
                                             name='HyponymHypernymOutput')(hidden_layer)
    model = tf.keras.Model(inputs=[input_word_ids, segment_ids], outputs=output_layer, name='BERT_CNN')
    model.build(input_shape=[(None, max_seq_len), (None, max_seq_len)])
    load_stock_weights(bert_layer, bert_model_ckpt)
    model.compile(optimizer=pf.optimizers.RAdam(learning_rate=learning_rate), loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.AUC(name='auc')], experimental_run_tf_function=not bayesian)
    return model


def train_neural_network(X_train: Tuple[np.ndarray, np.ndarray], y_train: np.ndarray,
                         X_val: Tuple[np.ndarray, np.ndarray], y_val: np.ndarray,
                         neural_network: tf.keras.Model, batch_size: int, max_epochs: int) -> tf.keras.Model:
    print('Structure of neural network:')
    print('')
    neural_network.summary()
    print('')
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=min(max_epochs, 2), monitor='val_auc', mode='max',
                                                  restore_best_weights=True, verbose=1)]
    neural_network.fit(X_train, y_train, epochs=max_epochs, verbose=1, callbacks=callbacks,
                       validation_data=(X_val, y_val), shuffle=True, batch_size=batch_size)
    print('')
    return neural_network


def evaluate_neural_network(X: Tuple[np.ndarray, np.ndarray], y: np.ndarray, neural_network: tf.keras.Model,
                            batch_size: int, num_monte_carlo: int = 0):
    assert isinstance(num_monte_carlo, int)
    assert num_monte_carlo >= 0
    if num_monte_carlo > 0:
        assert num_monte_carlo > 1
    probabilities = neural_network.predict(X, batch_size=batch_size)
    if num_monte_carlo > 0:
        for _ in range(num_monte_carlo - 1):
            probabilities += neural_network.predict(X, batch_size=batch_size)
        probabilities /= float(num_monte_carlo)
    probabilities = probabilities.reshape((max(probabilities.shape),))
    print('Evaluation results:')
    print('  ROC-AUC is   {0:.6f}'.format(roc_auc_score(y, probabilities)))
    y_pred = np.asarray(probabilities >= 0.5, dtype=np.uint8)
    del probabilities
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        print('  Precision is {0:.6f}'.format(precision_score(y, y_pred, average='binary')))
        print('  Recall is    {0:.6f}'.format(recall_score(y, y_pred, average='binary')))
        print('  F1 is        {0:.6f}'.format(f1_score(y, y_pred, average='binary')))
    print('')
    del y_pred


def do_submission(submission_result_name: str, neural_network: tf.keras.Model, bert_tokenizer: FullTokenizer,
                  max_seq_len: int, batch_size: int, input_hyponyms: List[tuple],
                  occurrences_of_input_hyponyms: List[Dict[str, List[Tuple[str, Tuple[int, int]]]]],
                  wordnet_synsets: Dict[str, List[str]], wordnet_source_senses: Dict[str, str],
                  wordnet_inflected_senses: Dict[str, Dict[str, Tuple[tuple, Tuple[int, int]]]],
                  num_monte_carlo: int = 0):
    n_data_parts = 50
    with codecs.open(submission_result_name, mode='w', encoding='utf-8', errors='ignore') as fp:
        data_writer = csv.writer(fp, delimiter='\t', quotechar='"')
        data_part_size = int(np.ceil(len(input_hyponyms) / n_data_parts))
        data_part_counter = 0
        for idx, hyponym in enumerate(input_hyponyms):
            contexts = tokenize_many_text_pairs_for_bert(
                generate_context_pairs_for_submission(
                    unseen_hyponym=hyponym, occurrences_of_hyponym=occurrences_of_input_hyponyms[idx],
                    synsets_with_sense_ids=wordnet_synsets, source_senses=wordnet_source_senses,
                    inflected_senses=wordnet_inflected_senses
                ),
                bert_tokenizer
            )
            if max_seq_len < MAX_SEQ_LENGTH:
                contexts = list(filter(lambda it: len(it[0]) <= max_seq_len, contexts))
            X = create_dataset_for_bert(text_pairs=contexts, seq_len=max_seq_len, batch_size=batch_size)
            assert isinstance(X, np.ndarray)
            probabilities = neural_network.predict(X, batch_size=batch_size)
            if num_monte_carlo > 0:
                for _ in range(num_monte_carlo - 1):
                    probabilities += neural_network.predict(X, batch_size=batch_size)
                probabilities /= float(num_monte_carlo)
            probabilities = probabilities.reshape((max(probabilities.shape),))
            del X
            assert probabilities.shape[0] == len(contexts)
            best_synsets = list(map(lambda idx: (contexts[idx][2], probabilities[idx]), range(len(contexts))))
            del contexts, probabilities
            best_synsets.sort(key=lambda it: (-it[1], it[0]))
            selected_synset_IDs = list()
            set_of_synset_IDs = set()
            for synset_id, proba in best_synsets:
                if synset_id not in set_of_synset_IDs:
                    set_of_synset_IDs.add(synset_id)
                    selected_synset_IDs.append(synset_id)
                if len(selected_synset_IDs) >= 10:
                    break
            del best_synsets
            for synset_id in selected_synset_IDs:
                data_writer.writerow([' '.join(hyponym).upper(), synset_id])
            if (idx + 1) % data_part_size == 0:
                data_part_counter += 1
                print('  {0} % of data for submission have been processed...'.format(data_part_counter * 2))
            del selected_synset_IDs, set_of_synset_IDs
        if data_part_counter < n_data_parts:
            print('  100 % of data for submission have been processed...')
