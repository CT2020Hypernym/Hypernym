import array
import codecs
import csv
import multiprocessing
import os
import random
from typing import Dict, List, Sequence, Tuple, Union
import warnings

from bert.tokenization.bert_tokenization import FullTokenizer
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.exceptions import UndefinedMetricWarning
import tensorflow_hub as hub
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
    n = int(round(0.95 * (len(lengths_of_texts) - 1)))
    if n > 0:
        optimal_length = 4
        while optimal_length < lengths_of_texts[n]:
            optimal_length *= 2
        optimal_length = min(optimal_length, MAX_SEQ_LENGTH)
        print('An optimal number of sub-tokens in the BERT input is {0}.'.format(optimal_length))
    else:
        optimal_length = MAX_SEQ_LENGTH
    return optimal_length


class BertDatasetGenerator(tf.keras.utils.Sequence):
    def __init__(self, text_pairs: List[Tuple[Sequence[int], int, Union[int, str]]], batch_size: int, seq_len: int):
        self.text_pairs = text_pairs
        self.batch_size = batch_size
        self.indices_of_samples = list(range(len(self.text_pairs)))
        self.seq_len = seq_len

    def shuffle_samples(self):
        random.shuffle(self.indices_of_samples)

    def __len__(self):
        return int(np.ceil(len(self.text_pairs) / float(self.batch_size)))

    def __getitem__(self, item):
        batch_start = item * self.batch_size
        batch_end = min(batch_start + self.batch_size, len(self.text_pairs))
        tokens = np.zeros((batch_end - batch_start, self.seq_len), dtype=np.int32)
        mask = np.zeros((batch_end - batch_start, self.seq_len), dtype=np.int32)
        segments = np.zeros((batch_end - batch_start, self.seq_len), dtype=np.int32)
        y = None
        for sample_idx in range(batch_start, batch_end):
            token_ids, n_left_tokens, additional_data = self.text_pairs[self.indices_of_samples[sample_idx]]
            for token_idx in range(len(token_ids)):
                tokens[sample_idx - batch_start][token_idx] = token_ids[token_idx]
                mask[sample_idx - batch_start][token_idx] = 1
                segments[sample_idx - batch_start][token_idx] = 1 if token_idx < n_left_tokens else 0
            assert isinstance(additional_data, int) or isinstance(additional_data, str)
            if isinstance(additional_data, int):
                if y is None:
                    y = [additional_data]
                else:
                    y.append(additional_data)
        if y is None:
            return tokens, mask, segments
        assert len(y) == (batch_end - batch_start)
        return (tokens, mask, segments), np.array(y, dtype=np.int32)


def build_simple_bert(model_name: str, optimal_seq_len: int = None) -> Tuple[FullTokenizer, tf.keras.Model]:
    if optimal_seq_len is None:
        seq_len = MAX_SEQ_LENGTH
    else:
        assert (optimal_seq_len > 0) and (optimal_seq_len <= MAX_SEQ_LENGTH)
        seq_len = optimal_seq_len
    input_word_ids = tf.keras.layers.Input(shape=(seq_len,), dtype=tf.int32, name="input_word_ids_for_BERT")
    input_mask = tf.keras.layers.Input(shape=(seq_len,), dtype=tf.int32, name="input_mask_for_BERT")
    segment_ids = tf.keras.layers.Input(shape=(seq_len,), dtype=tf.int32, name="segment_ids_for_BERT")
    bert_layer = hub.KerasLayer(model_name, trainable=False, name='BERT_Layer')
    pooled_output, _ = bert_layer([input_word_ids, input_mask, segment_ids])
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = FullTokenizer(vocab_file, do_lower_case)
    output_layer = tf.keras.layers.Dense(units=1, activation='sigmoid', kernel_initializer='glorot_uniform',
                                         name='HyponymHypernymOutput')(pooled_output)
    model = tf.keras.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=output_layer, name='SimpleBERT')
    return tokenizer, model


def build_bert_and_cnn(model_name: str, n_filters: int, hidden_layer_size: int, ave_pooling: bool, bayesian: bool,
                       optimal_seq_len: int = None, **kwargs) -> Tuple[FullTokenizer, tf.keras.Model]:
    if optimal_seq_len is None:
        seq_len = MAX_SEQ_LENGTH
    else:
        assert (optimal_seq_len > 0) and (optimal_seq_len <= MAX_SEQ_LENGTH)
        seq_len = optimal_seq_len
    if bayesian:
        assert 'kl_weight' in kwargs
    input_word_ids = tf.keras.layers.Input(shape=(seq_len,), dtype=tf.int32, name="input_word_ids_for_BERT")
    input_mask = tf.keras.layers.Input(shape=(seq_len,), dtype=tf.int32, name="input_mask_for_BERT")
    segment_ids = tf.keras.layers.Input(shape=(seq_len,), dtype=tf.int32, name="segment_ids_for_BERT")
    bert_layer = hub.KerasLayer(model_name, trainable=False, name='BERT_Layer')
    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = FullTokenizer(vocab_file, do_lower_case)
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
                                            name='Conv_1grams')(sequence_output),
            tfp.layers.Convolution1DFlipout(kernel_size=2, filters=n_filters, activation=activation_type,
                                            kernel_divergence_fn=kl_divergence_function, padding='same',
                                            name='Conv_2grams')(sequence_output),
            tfp.layers.Convolution1DFlipout(kernel_size=3, filters=n_filters, activation=activation_type,
                                            kernel_divergence_fn=kl_divergence_function, padding='same',
                                            name='Conv_3grams')(sequence_output),
            tfp.layers.Convolution1DFlipout(kernel_size=4, filters=n_filters, activation=activation_type,
                                            kernel_divergence_fn=kl_divergence_function, padding='same',
                                            name='Conv_4grams')(sequence_output),
            tfp.layers.Convolution1DFlipout(kernel_size=5, filters=n_filters, activation=activation_type,
                                            kernel_divergence_fn=kl_divergence_function, padding='same',
                                            name='Conv_5grams')(sequence_output),
        ]
    else:
        kl_divergence_function = None
        conv_layers = [
            tf.keras.layers.Conv1D(kernel_size=1, filters=n_filters, activation=activation_type, padding='same',
                                   kernel_initializer=initializer_type, name='Conv_1grams')(sequence_output),
            tf.keras.layers.Conv1D(kernel_size=2, filters=n_filters, activation=activation_type, padding='same',
                                   kernel_initializer=initializer_type, name='Conv_2grams')(sequence_output),
            tf.keras.layers.Conv1D(kernel_size=3, filters=n_filters, activation=activation_type, padding='same',
                                   kernel_initializer=initializer_type, name='Conv_3grams')(sequence_output),
            tf.keras.layers.Conv1D(kernel_size=4, filters=n_filters, activation=activation_type, padding='same',
                                   kernel_initializer=initializer_type, name='Conv_4grams')(sequence_output),
            tf.keras.layers.Conv1D(kernel_size=5, filters=n_filters, activation=activation_type, padding='same',
                                   kernel_initializer=initializer_type, name='Conv_5grams')(sequence_output)
        ]
    conv_concat_layer = tf.keras.layers.Concatenate(name='ConvConcat')(conv_layers)
    if ave_pooling:
        masking_calc = MaskCalculator(output_dim=len(conv_layers) * n_filters, trainable=False,
                                      name='MaskCalculator')(input_mask)
        conv_concat_layer = tf.keras.layers.Multiply(name='MaskMultiplicator')([conv_concat_layer, masking_calc])
        conv_concat_layer = tf.keras.layers.Masking(name='Masking')(conv_concat_layer)
        feature_layer = tf.keras.layers.Concatenate(name='FeatureLayer')(
            [pooled_output, tf.keras.layers.GlobalAveragePooling1D(name='AvePooling')(conv_concat_layer)]
        )
    else:
        feature_layer = tf.keras.layers.Concatenate(name='FeatureLayer')(
            [pooled_output, tf.keras.layers.GlobalMaxPooling1D(name='MaxPooling')(conv_concat_layer)]
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
    model = tf.keras.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=output_layer, name='BERT_CNN')
    return tokenizer, model


def train_neural_network(data_for_training: BertDatasetGenerator, data_for_validation: BertDatasetGenerator,
                         neural_network: tf.keras.Model, max_epochs: int, learning_rate: float,
                         is_bayesian: bool) -> tf.keras.Model:
    print('Structure of neural network:')
    print('')
    neural_network.summary()
    print('')
    neural_network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                           loss=tf.keras.losses.BinaryCrossentropy(),
                           metrics=[tf.keras.metrics.AUC(name='auc')], experimental_run_tf_function=not is_bayesian)
    print('')
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=min(max_epochs, 3), monitor='val_auc', mode='max',
                                                  restore_best_weights=True, verbose=1)]
    neural_network.fit(data_for_training, epochs=max_epochs, verbose=1, callbacks=callbacks,
                       validation_data=data_for_validation, shuffle=True)
    print('')
    return neural_network


def evaluate_neural_network(dataset: BertDatasetGenerator, neural_network: tf.keras.Model, num_monte_carlo: int = 0):
    assert isinstance(num_monte_carlo, int)
    assert num_monte_carlo >= 0
    if num_monte_carlo > 0:
        assert num_monte_carlo > 1
    probabilities = neural_network.predict(dataset)
    if num_monte_carlo > 0:
        for _ in range(num_monte_carlo - 1):
            probabilities += neural_network.predict(dataset)
        probabilities /= float(num_monte_carlo)
    probabilities = probabilities.reshape((max(probabilities.shape),))
    y_true = []
    for _, batch_y in dataset:
        y_true.append(batch_y)
    y_true = np.concatenate(y_true)
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


def apply_neural_network(dataset: BertDatasetGenerator, neural_network: tf.keras.Model,
                         num_monte_carlo: int = 0) -> np.ndarray:
    assert isinstance(num_monte_carlo, int)
    assert num_monte_carlo >= 0
    if num_monte_carlo > 0:
        assert num_monte_carlo > 1
    probabilities = neural_network.predict(dataset)
    if num_monte_carlo > 0:
        for _ in range(num_monte_carlo - 1):
            probabilities += neural_network.predict(dataset)
        probabilities /= float(num_monte_carlo)
    probabilities = probabilities.reshape((max(probabilities.shape),))
    return probabilities


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
            dataset_generator = BertDatasetGenerator(text_pairs=contexts, batch_size=batch_size, seq_len=max_seq_len)
            probabilities = apply_neural_network(
                dataset=dataset_generator, neural_network=neural_network,
                num_monte_carlo=num_monte_carlo
            )
            del dataset_generator
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
