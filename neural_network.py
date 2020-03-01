import os
import tempfile
import time
from typing import Tuple
import warnings

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.exceptions import UndefinedMetricWarning
import tensorflow as tf
import tensorflow_probability as tfp

from trainset_preparing import TrainsetGenerator


class MaskCalculator(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MaskCalculator, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MaskCalculator, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.keras.backend.permute_dimensions(
            x=tf.keras.backend.repeat(
                x=tf.keras.backend.cast(
                    x=tf.keras.backend.greater(
                        x=inputs,
                        y=0
                    ),
                    dtype='float32'
                ),
                n=self.output_dim
            ),
            pattern=(0, 2, 1)
        )

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 1
        shape = list(input_shape)
        shape.append(self.output_dim)
        return tuple(shape)


def calculate_learning_rate(epoch_index: int, cycle_length: int, min_lr: float, max_lr: float) -> float:
    assert cycle_length > 3
    assert min_lr > 0.0
    assert max_lr > min_lr
    assert epoch_index >= 0
    cycle = 1 + epoch_index // cycle_length
    time_in_cycle = epoch_index % cycle_length
    h = (max_lr - min_lr) / float(cycle)
    if time_in_cycle <= (cycle_length // 2):
        lr = min_lr + h * time_in_cycle / float(cycle_length // 2)
    else:
        lr = min_lr + h * (cycle_length - 1 - time_in_cycle) / float(cycle_length // 2)
    return lr


def build_cnn(max_hyponym_length: int, max_hypernym_length: int, word_embeddings: np.ndarray,
              n_feature_maps: int, hidden_layer_size: int, n_hidden_layers: int, dropout_rate: float) -> tf.keras.Model:
    EPS = 1e-4
    if dropout_rate > EPS:
        assert dropout_rate < (1.0 - EPS)
    hyponym_text = tf.keras.layers.Input(name='HyponymInput', shape=(max_hyponym_length,), dtype='int32')
    hypernym_text = tf.keras.layers.Input(name='HypernymInput', shape=(max_hypernym_length,), dtype='int32')
    hyponym_embedding_layer = tf.keras.layers.Embedding(
        input_dim=word_embeddings.shape[0], output_dim=word_embeddings.shape[1], input_length=max_hyponym_length,
        weights=[word_embeddings], trainable=False, name='HyponymEmbedding'
    )(hyponym_text)
    hypernym_embedding_layer = tf.keras.layers.Embedding(
        input_dim=word_embeddings.shape[0], output_dim=word_embeddings.shape[1], input_length=max_hypernym_length,
        weights=[word_embeddings], trainable=False, name='HypernymEmbedding'
    )(hypernym_text)
    hyponym_masking_calc = MaskCalculator(output_dim=2 * n_feature_maps, trainable=False,
                                          name='HyponymMaskCalculator')(hyponym_text)
    hypernym_masking_calc = MaskCalculator(output_dim=3 * n_feature_maps, trainable=False,
                                           name='HypernymMaskCalculator')(hypernym_text)
    conv_layers_for_hyponym = [
        tf.keras.layers.Conv1D(
            filters=n_feature_maps, kernel_size=1, activation='tanh', name='HyponymUnigramsConv',
            kernel_initializer="glorot_uniform", padding='same'
        )(hyponym_embedding_layer),
        tf.keras.layers.Conv1D(
            filters=n_feature_maps, kernel_size=2, activation='tanh', name='HyponymBigramsConv',
            kernel_initializer="glorot_uniform", padding='same'
        )(hyponym_embedding_layer)
    ]
    concat_conv_layer_for_hyponym = tf.keras.layers.Concatenate(name='HyponymConcat')(conv_layers_for_hyponym)
    concat_conv_layer_for_hyponym = tf.keras.layers.Multiply(
        name='HyponymMaskMultiplicator', trainable=False
    )([concat_conv_layer_for_hyponym, hyponym_masking_calc])
    concat_conv_layer_for_hyponym = tf.keras.layers.Masking(name='HyponymMasking')(concat_conv_layer_for_hyponym)
    pooled_layer_for_hyponym = tf.keras.layers.GlobalAveragePooling1D(
        name='HyponymAvePooling'
    )(concat_conv_layer_for_hyponym)
    conv_layers_for_hypernym = [
        tf.keras.layers.Conv1D(
            filters=n_feature_maps, kernel_size=1, activation='tanh', name='HypernymUnigramsConv',
            kernel_initializer="glorot_uniform", padding='same'
        )(hypernym_embedding_layer),
        tf.keras.layers.Conv1D(
            filters=n_feature_maps, kernel_size=2, activation='tanh', name='HypernymBigramsConv',
            kernel_initializer="glorot_uniform", padding='same'
        )(hypernym_embedding_layer),
        tf.keras.layers.Conv1D(
            filters=n_feature_maps, kernel_size=3, activation='tanh', name='HypernymTrigramsConv',
            kernel_initializer="glorot_uniform", padding='same'
        )(hypernym_embedding_layer)
    ]
    concat_conv_layer_for_hypernym = tf.keras.layers.Concatenate(name='HypernymConcat')(conv_layers_for_hypernym)
    concat_conv_layer_for_hypernym = tf.keras.layers.Multiply(
        name='HypernymMaskMultiplicator', trainable=False
    )([concat_conv_layer_for_hypernym, hypernym_masking_calc])
    concat_conv_layer_for_hypernym = tf.keras.layers.Masking(name='HypernymMasking')(concat_conv_layer_for_hypernym)
    pooled_layers_for_hypernym = tf.keras.layers.GlobalAveragePooling1D(
        name='HypernymAvePooling'
    )(concat_conv_layer_for_hypernym)
    concatenated = tf.keras.layers.Concatenate(name='ConcatAll')([pooled_layer_for_hyponym, pooled_layers_for_hypernym])
    if dropout_rate > EPS:
        concatenated = tf.keras.layers.Dropout(rate=dropout_rate, name='Dropout0')(concatenated)
    hidden_layer = tf.keras.layers.Dense(
        units=hidden_layer_size, activation='tanh', name='HiddenLayer1', kernel_initializer="glorot_uniform"
    )(concatenated)
    if dropout_rate > EPS:
        hidden_layer = tf.keras.layers.Dropout(rate=dropout_rate, name='Dropout1')(hidden_layer)
    for layer_idx in range(1, n_hidden_layers):
        hidden_layer = tf.keras.layers.Dense(
            units=hidden_layer_size, activation='tanh', name='HiddenLayer{0}'.format(layer_idx + 1),
            kernel_initializer="glorot_uniform"
        )(hidden_layer)
        if dropout_rate > EPS:
            hidden_layer = tf.keras.layers.Dropout(rate=dropout_rate,
                                                   name='Dropout{0}'.format(layer_idx + 1))(hidden_layer)
    output_layer = tf.keras.layers.Dense(
        units=1, activation='sigmoid', name='OutputLayer',
        kernel_initializer="glorot_uniform"
    )(hidden_layer)
    neural_network = tf.keras.Model(inputs=[hyponym_text, hypernym_text], outputs=output_layer, name='ConvNN')
    return neural_network


def build_bayesian_cnn(max_hyponym_length: int, max_hypernym_length: int, word_embeddings: np.ndarray,
                       n_feature_maps: int, hidden_layer_size: int, n_hidden_layers: int,
                       n_train_samples: int, kl_weight: float) -> tf.keras.Model:
    hyponym_text = tf.keras.layers.Input(name='HyponymInput', shape=(max_hyponym_length,), dtype='int32')
    hypernym_text = tf.keras.layers.Input(name='HypernymInput', shape=(max_hypernym_length,), dtype='int32')
    hyponym_embedding_layer = tf.keras.layers.Embedding(
        input_dim=word_embeddings.shape[0], output_dim=word_embeddings.shape[1], input_length=max_hyponym_length,
        weights=[word_embeddings], trainable=False, name='HyponymEmbedding'
    )(hyponym_text)
    hypernym_embedding_layer = tf.keras.layers.Embedding(
        input_dim=word_embeddings.shape[0], output_dim=word_embeddings.shape[1], input_length=max_hypernym_length,
        weights=[word_embeddings], trainable=False, name='HypernymEmbedding'
    )(hypernym_text)
    hyponym_masking_calc = MaskCalculator(output_dim=2 * n_feature_maps, trainable=False,
                                          name='HyponymMaskCalculator')(hyponym_text)
    hypernym_masking_calc = MaskCalculator(output_dim=3 * n_feature_maps, trainable=False,
                                           name='HypernymMaskCalculator')(hypernym_text)
    kl_divergence_function = (
        lambda q, p, _: (tfp.distributions.kl_divergence(q, p) * tf.constant(kl_weight / float(n_train_samples),
                                                                             dtype=tf.float32, name='KL_weight'))
    )
    conv_layers_for_hyponym = [
        tfp.layers.Convolution1DFlipout(
            filters=n_feature_maps, kernel_size=1, activation='tanh', name='HyponymUnigramsConv', padding='same',
            kernel_divergence_fn=kl_divergence_function
        )(hyponym_embedding_layer),
        tfp.layers.Convolution1DFlipout(
            filters=n_feature_maps, kernel_size=2, activation='tanh', name='HyponymBigramsConv', padding='same',
            kernel_divergence_fn=kl_divergence_function
        )(hyponym_embedding_layer)
    ]
    concat_conv_layer_for_hyponym = tf.keras.layers.Concatenate(name='HyponymConcat')(conv_layers_for_hyponym)
    concat_conv_layer_for_hyponym = tf.keras.layers.Multiply(
        name='HyponymMaskMultiplicator', trainable=False
    )([concat_conv_layer_for_hyponym, hyponym_masking_calc])
    concat_conv_layer_for_hyponym = tf.keras.layers.Masking(name='HyponymMasking')(concat_conv_layer_for_hyponym)
    pooled_layer_for_hyponym = tf.keras.layers.GlobalAveragePooling1D(
        name='HyponymAvePooling'
    )(concat_conv_layer_for_hyponym)
    conv_layers_for_hypernym = [
        tfp.layers.Convolution1DFlipout(
            filters=n_feature_maps, kernel_size=1, activation='tanh', name='HypernymUnigramsConv', padding='same',
            kernel_divergence_fn=kl_divergence_function
        )(hypernym_embedding_layer),
        tfp.layers.Convolution1DFlipout(
            filters=n_feature_maps, kernel_size=2, activation='tanh', name='HypernymBigramsConv', padding='same',
            kernel_divergence_fn=kl_divergence_function
        )(hypernym_embedding_layer),
        tfp.layers.Convolution1DFlipout(
            filters=n_feature_maps, kernel_size=3, activation='tanh', name='HypernymTrigramsConv', padding='same',
            kernel_divergence_fn=kl_divergence_function
        )(hypernym_embedding_layer)
    ]
    concat_conv_layer_for_hypernym = tf.keras.layers.Concatenate(name='HypernymConcat')(conv_layers_for_hypernym)
    concat_conv_layer_for_hypernym = tf.keras.layers.Multiply(
        name='HypernymMaskMultiplicator', trainable=False
    )([concat_conv_layer_for_hypernym, hypernym_masking_calc])
    concat_conv_layer_for_hypernym = tf.keras.layers.Masking(name='HypernymMasking')(concat_conv_layer_for_hypernym)
    pooled_layers_for_hypernym = tf.keras.layers.GlobalAveragePooling1D(
        name='HypernymAvePooling'
    )(concat_conv_layer_for_hypernym)
    concatenated = tf.keras.layers.Concatenate(name='ConcatAll')([pooled_layer_for_hyponym, pooled_layers_for_hypernym])
    hidden_layer = tfp.layers.DenseFlipout(
        units=hidden_layer_size, activation='tanh', name='HiddenLayer1',
        kernel_divergence_fn=kl_divergence_function
    )(concatenated)
    for layer_idx in range(1, n_hidden_layers):
        hidden_layer = tfp.layers.DenseFlipout(
            units=hidden_layer_size, activation='tanh', name='HiddenLayer{0}'.format(layer_idx + 1),
            kernel_divergence_fn=kl_divergence_function
        )(hidden_layer)
    output_layer = tfp.layers.DenseFlipout(
        units=1, activation='sigmoid', name='OutputLayer',
        kernel_divergence_fn=kl_divergence_function
    )(hidden_layer)
    neural_network = tf.keras.Model(inputs=[hyponym_text, hypernym_text], outputs=output_layer, name='BayesianConvNN')
    return neural_network


def train_neural_network(data_for_training: TrainsetGenerator, data_for_validation: TrainsetGenerator,
                         neural_network: tf.keras.Model, is_bayesian: bool,
                         training_cycle_length: int, min_learning_rate: float, max_learning_rate: float,
                         max_epochs: int, **kwargs) -> tf.keras.Model:
    assert training_cycle_length > 0
    assert training_cycle_length < max_epochs
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
        for epoch in range(max_epochs):
            epoch_accuracy = 0.0
            epoch_loss = 0.0
            neural_network.reset_metrics()
            start_time = time.time()
            tf.keras.backend.set_value(
                neural_network.optimizer.lr,
                calculate_learning_rate(
                    epoch_index=epoch,
                    cycle_length=training_cycle_length,
                    max_lr=max_learning_rate, min_lr=min_learning_rate
                )
            )
            for iter_idx, (batch_x, batch_y) in enumerate(data_for_training):
                epoch_loss, epoch_accuracy = neural_network.train_on_batch(batch_x, batch_y, reset_metrics=False)
            training_duration = time.time() - start_time
            print("Epoch {0}".format(epoch + 1))
            print("  Training time is {0:.3f} secs".format(training_duration))
            print("  Learning rate is {0:.7f}".format(float(tf.keras.backend.get_value(neural_network.optimizer.lr))))
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
                    accuracy_score(y_true, y_pred), roc_auc, f1_score(y_true, y_pred), precision_score(y_true, y_pred),
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


def evaluate_neural_network(X: Tuple[np.ndarray, np.ndarray], y_true: np.ndarray, neural_network: tf.keras.Model,
                            batch_size: int, num_monte_carlo: int = 0):
    assert isinstance(X, tuple) or isinstance(X, list)
    assert len(X) == 2
    assert isinstance(X[0], np.ndarray)
    assert isinstance(X[1], np.ndarray)
    assert isinstance(y_true, np.ndarray)
    assert len(X[0].shape) == 2
    assert len(X[1].shape) == 2
    assert len(y_true.shape) == 1
    assert X[0].shape[0] == X[1].shape[0]
    assert X[0].shape[0] == y_true.shape[0]
    probabilities = apply_neural_network(X, neural_network, batch_size, num_monte_carlo)
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


def apply_neural_network(X: Tuple[np.ndarray, np.ndarray], neural_network: tf.keras.Model,
                         batch_size: int, num_monte_carlo: int = 0) -> np.ndarray:
    assert isinstance(X, tuple) or isinstance(X, list)
    assert len(X) == 2
    assert isinstance(X[0], np.ndarray)
    assert isinstance(X[1], np.ndarray)
    assert len(X[0].shape) == 2
    assert len(X[1].shape) == 2
    assert X[0].shape[0] == X[1].shape[0]
    if num_monte_carlo > 0:
        probs = tf.stack([neural_network.predict(X) for _ in range(num_monte_carlo)], axis=0)
        probabilities = tf.reduce_mean(probs, axis=0)
        del probs
    else:
        probabilities = neural_network.predict(X, batch_size=batch_size)
    n = len(probabilities.shape)
    assert (n <= 2) and (n > 0)
    if n == 1:
        return probabilities
    assert 1 in set(probabilities.shape)
    return np.reshape(probabilities, newshape=(max(probabilities.shape),))
