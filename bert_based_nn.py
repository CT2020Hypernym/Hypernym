import os
import random
import tempfile
import time
from typing import List, Tuple, Union
import warnings

from bert.tokenization.bert_tokenization import FullTokenizer
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.exceptions import UndefinedMetricWarning
import tensorflow_hub as hub
import tensorflow as tf

from neural_network import calculate_learning_rate


MAX_SEQ_LENGTH = 512


def tokenize_text_pair_for_bert(left_text: str, right_text: str, bert_tokenizer: FullTokenizer) -> Tuple[List[int],
                                                                                                         List[int]]:
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
    start_of_left_text = 1
    end_of_left_text = len(tokenized_left_text) - 1
    start_of_right_text = end_of_left_text + 1
    end_of_right_text = len(tokenized_text)
    if len(tokenized_text) > MAX_SEQ_LENGTH:
        warnings.warn("The text pair `{0}` - `{1}` contains too many sub-tokens!".format(left_text, right_text))
        while len(tokenized_text) > MAX_SEQ_LENGTH:
            if random.random() >= 0.5:
                removed_token_idx = random.randint(start_of_right_text, end_of_right_text)
                tokenized_text = tokenized_text[0:removed_token_idx] + tokenized_text[(removed_token_idx + 1):]
                end_of_right_text -= 1
            else:
                removed_token_idx = random.randint(start_of_left_text, end_of_left_text)
                tokenized_text = tokenized_text[0:removed_token_idx] + tokenized_text[(removed_token_idx + 1):]
                end_of_left_text -= 1
                start_of_right_text -= 1
                end_of_right_text -= 1
    token_IDs = bert_tokenizer.convert_tokens_to_ids(tokenized_text)
    segment_IDs = [1 for _ in range(end_of_left_text + 1)] + [0 for _ in range(end_of_right_text - start_of_right_text)]
    assert len(token_IDs) == len(segment_IDs)
    return token_IDs, segment_IDs


class BertDatasetGenerator(tf.keras.utils.Sequence):
    def __init__(self, text_pairs: List[Tuple[str, str, Union[int, str]]], tokenizer: FullTokenizer, batch_size: int):
        self.text_pairs = text_pairs
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.indices_of_samples = list(range(len(self.text_pairs)))

    def shuffle_samples(self):
        random.shuffle(self.indices_of_samples)

    def __len__(self):
        return int(np.ceil(len(self.text_pairs) / float(self.batch_size)))

    def __getitem__(self, item):
        batch_start = item * self.batch_size
        batch_end = min(batch_start + self.batch_size, len(self.texts))
        tokens = np.zeros((batch_end - batch_start, MAX_SEQ_LENGTH), dtype=np.int32)
        mask = np.zeros((batch_end - batch_start, MAX_SEQ_LENGTH), dtype=np.int32)
        segments = np.zeros((batch_end - batch_start, MAX_SEQ_LENGTH), dtype=np.int32)
        y = None
        for sample_idx in range(batch_start, batch_end):
            token_ids, segment_ids = tokenize_text_pair_for_bert(
                left_text=self.text_pairs[self.indices_of_samples[sample_idx]][0],
                right_text=self.text_pairs[self.indices_of_samples[sample_idx]][1],
                bert_tokenizer=self.tokenizer
            )
            for token_idx in range(len(token_ids)):
                tokens[sample_idx - batch_start][token_idx] = token_ids[token_idx]
                mask[sample_idx - batch_start][token_idx] = 1
                segments[sample_idx - batch_start][token_idx] = segment_ids[token_idx]
            assert isinstance(self.text_pairs[self.indices_of_samples[sample_idx]][2], int) or \
                   isinstance(self.text_pairs[self.indices_of_samples[sample_idx]][2], str)
            if isinstance(self.text_pairs[sample_idx][2], int):
                if y is None:
                    y = [self.text_pairs[self.indices_of_samples[sample_idx]][2]]
                else:
                    y.append(self.text_pairs[self.indices_of_samples[sample_idx]][2])
        if y is None:
            return tokens, mask, segments
        assert len(y) == (batch_end - batch_start)
        return (tokens, mask, segments), np.array(y, dtype=np.int32)


def build_simple_bert(model_name: str) -> Tuple[FullTokenizer, tf.keras.Model]:
    input_word_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LENGTH,), dtype=tf.int32, name="input_word_ids_for_BERT")
    input_mask = tf.keras.layers.Input(shape=(MAX_SEQ_LENGTH,), dtype=tf.int32, name="input_mask_for_BERT")
    segment_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LENGTH,), dtype=tf.int32, name="segment_ids_for_BERT")
    bert_layer = hub.KerasLayer(model_name, trainable=True)
    pooled_output, _ = bert_layer([input_word_ids, input_mask, segment_ids])
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = FullTokenizer(vocab_file, do_lower_case)
    output_layer = tf.keras.layers.Dense(units=1, activation='sigmoid', kernel_initializer='glorot_uniform',
                                         name='HyponymHypernymOutput')(pooled_output)
    model = tf.keras.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=output_layer)
    return tokenizer, model


def train_neural_network(data_for_training: BertDatasetGenerator, data_for_validation: BertDatasetGenerator,
                         neural_network: tf.keras.Model, is_bayesian: bool,
                         training_cycle_length: int, min_learning_rate: float, max_learning_rate: float,
                         max_iters: int, eval_every: int, **kwargs) -> tf.keras.Model:
    assert training_cycle_length > 0
    assert training_cycle_length < max_iters
    patience = training_cycle_length * 2
    if is_bayesian:
        assert 'num_monte_carlo' in kwargs
        assert isinstance(kwargs['num_monte_carlo'], int)
        assert kwargs['num_monte_carlo'] > 1
    print('Structure of neural network:')
    print('')
    neural_network.summary()
    print('')
    neural_network.compile(optimizer=tf.keras.optimizers.Adam(min_learning_rate),
                           loss=tf.keras.losses.BinaryCrossentropy(),
                           metrics=["binary_accuracy"], experimental_run_tf_function=not is_bayesian)
    temp_file_name = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as fp:
            temp_file_name = fp.name
        best_auc = None
        epochs_without_improving = 0
        n_train_batches = len(data_for_training)
        start_time = time.time()
        for iter in range(max_iters):
            batch_idx = iter % n_train_batches
            batch_x, batch_y = data_for_training[batch_idx]
            epoch_loss, epoch_accuracy = neural_network.train_on_batch(batch_x, batch_y, reset_metrics=False)
            if batch_idx == (n_train_batches - 1):
                data_for_training.shuffle_samples()
            if (batch_idx == (n_train_batches - 1)) or ((iter % eval_every) == 0):
                training_duration = time.time() - start_time
                print("Iteration {0}".format(iter + 1))
                print("  Training time is {0:.3f} secs".format(training_duration))
                print(
                    "  Learning rate is {0:.7f}".format(float(tf.keras.backend.get_value(neural_network.optimizer.lr))))
                print("  Training measures:")
                print("    loss = {0:.6f}, accuracy = {1:8.6f}".format(epoch_loss, epoch_accuracy))
                y_true = []
                probabilities = []
                start_time = time.time()
                for batch_x, batch_y in data_for_validation:
                    if is_bayesian:
                        batch_predicted = tf.reduce_mean(
                            [neural_network.predict_on_batch(batch_x) for _ in range(kwargs['num_monte_carlo'])],
                            axis=0
                        )
                    else:
                        batch_predicted = neural_network.predict_on_batch(batch_x)
                    y_true.append(batch_y)
                    if not isinstance(batch_predicted, np.ndarray):
                        batch_predicted = batch_predicted.numpy()
                    probabilities.append(batch_predicted.reshape(batch_y.shape))
                    del batch_predicted
                validation_duration = time.time() - start_time
                y_true = np.concatenate(y_true)
                probabilities = np.concatenate(probabilities)
                roc_auc = roc_auc_score(y_true, probabilities)
                y_pred = np.asarray(probabilities >= 0.5, dtype=y_true.dtype)
                print("  Validation time is {0:.3f} secs".format(validation_duration))
                print("  Validation measures:")
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                    print("    accuracy = {0:8.6f}, AUC = {1:8.6f}, F1 = {2:.8f}, P = {3:.8f}, R = {4:.8f}".format(
                        accuracy_score(y_true, y_pred), roc_auc, f1_score(y_true, y_pred),
                        precision_score(y_true, y_pred),
                        recall_score(y_true, y_pred)
                    ))
                if best_auc is None:
                    best_auc = roc_auc
                    neural_network.save_weights(temp_file_name, overwrite=True)
                else:
                    if roc_auc > best_auc:
                        best_auc = roc_auc
                        neural_network.save_weights(temp_file_name, overwrite=True)
                        epochs_without_improving = 0
                    else:
                        epochs_without_improving += 1
                    if epochs_without_improving > patience:
                        print("Early stopping!")
                        break
                start_time = time.time()
                neural_network.reset_metrics()
            tf.keras.backend.set_value(
                neural_network.optimizer.lr,
                calculate_learning_rate(epoch_index=iter, cycle_length=training_cycle_length,
                                        max_lr=max_learning_rate, min_lr=min_learning_rate)
            )
        if epochs_without_improving <= patience:
            print("Maximal number of epochs is reached!")
        print('')
        neural_network.load_weights(temp_file_name)
    finally:
        if temp_file_name is not None:
            if os.path.isfile(temp_file_name):
                os.remove(temp_file_name)
    print('')
    return neural_network


def evaluate_neural_network(dataset: BertDatasetGenerator, neural_network: tf.keras.Model, num_monte_carlo: int = 0):
    y_true = []
    probabilities = []
    for batch_x, batch_y in dataset:
        if num_monte_carlo > 0:
            batch_predicted = tf.reduce_mean(
                [neural_network.predict_on_batch(batch_x) for _ in range(num_monte_carlo)],
                axis=0
            )
        else:
            batch_predicted = neural_network.predict_on_batch(batch_x)
        y_true.append(batch_y)
        if not isinstance(batch_predicted, np.ndarray):
            batch_predicted = batch_predicted.numpy()
        probabilities.append(batch_predicted.reshape(batch_y.shape))
        del batch_predicted
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


def apply_neural_network(dataset: BertDatasetGenerator, neural_network: tf.keras.Model,
                         num_monte_carlo: int = 0) -> np.ndarray:
    probabilities = []
    for batch_x in dataset:
        if num_monte_carlo > 0:
            batch_predicted = tf.reduce_mean(
                [neural_network.predict_on_batch(batch_x) for _ in range(num_monte_carlo)],
                axis=0
            )
        else:
            batch_predicted = neural_network.predict_on_batch(batch_x)
        if not isinstance(batch_predicted, np.ndarray):
            batch_predicted = batch_predicted.numpy()
        probabilities.append(batch_predicted.reshape((max(batch_predicted.shape), )))
        del batch_predicted
    probabilities = np.concatenate(probabilities)
    return np.concatenate(probabilities)
