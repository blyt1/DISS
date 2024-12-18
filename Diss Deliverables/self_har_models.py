import tensorflow as tf
import numpy as np

__author__ = "C. I. Tang"
__copyright__ = "Copyright (C) 2021 C. I. Tang"
"""
This code was taken from SelfHAR with additions to the code
"""


"""
Complementing the work of Tang et al.: SelfHAR: Improving Human Activity Recognition through Self-training with Unlabeled Data
@article{10.1145/3448112,
  author = {Tang, Chi Ian and Perez-Pozuelo, Ignacio and Spathis, Dimitris and Brage, Soren and Wareham, Nick and Mascolo, Cecilia},
  title = {SelfHAR: Improving Human Activity Recognition through Self-Training with Unlabeled Data},
  year = {2021},
  issue_date = {March 2021},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  volume = {5},
  number = {1},
  url = {https://doi.org/10.1145/3448112},
  doi = {10.1145/3448112},
  abstract = {Machine learning and deep learning have shown great promise in mobile sensing applications, including Human Activity Recognition. However, the performance of such models in real-world settings largely depends on the availability of large datasets that captures diverse behaviors. Recently, studies in computer vision and natural language processing have shown that leveraging massive amounts of unlabeled data enables performance on par with state-of-the-art supervised models.In this work, we present SelfHAR, a semi-supervised model that effectively learns to leverage unlabeled mobile sensing datasets to complement small labeled datasets. Our approach combines teacher-student self-training, which distills the knowledge of unlabeled and labeled datasets while allowing for data augmentation, and multi-task self-supervision, which learns robust signal-level representations by predicting distorted versions of the input.We evaluated SelfHAR on various HAR datasets and showed state-of-the-art performance over supervised and previous semi-supervised approaches, with up to 12% increase in F1 score using the same number of model parameters at inference. Furthermore, SelfHAR is data-efficient, reaching similar performance using up to 10 times less labeled data compared to supervised approaches. Our work not only achieves state-of-the-art performance in a diverse set of HAR datasets, but also sheds light on how pre-training tasks may affect downstream performance.},
  journal = {Proc. ACM Interact. Mob. Wearable Ubiquitous Technol.},
  month = mar,
  articleno = {36},
  numpages = {30},
  keywords = {semi-supervised training, human activity recognition, unlabeled data, self-supervised training, self-training, deep learning}
}

Access to Article:
    https://doi.org/10.1145/3448112
    https://dl.acm.org/doi/abs/10.1145/3448112

Contact: cit27@cl.cam.ac.uk

Copyright (C) 2021 C. I. Tang

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
TODO: add github link
"""

def create_1d_conv_core_model(input_shape, model_name="base_model", use_standard_max_pooling=False):
    """
    Create the base model for activity recognition
    Reference (TPN model):
        Saeed, A., Ozcelebi, T., & Lukkien, J. (2019). Multi-task self-supervised learning for human activity detection. Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, 3(2), 1-30.
    Architecture:
        Input
        -> Conv 1D: 32 filters, 24 kernel_size, relu, L2 regularizer
        -> Dropout: 10%
        -> Conv 1D: 64 filters, 16 kernel_size, relu, L2 regularizer
        -> Dropout: 10%
        -> Conv 1D: 96 filters, 8 kernel_size, relu, L2 regularizer
        -> Dropout: 10%
        -> Global Maximum Pooling 1D
    
    Parameters:
        input_shape
            the input shape for the model, should be (window_size, num_channels)
    
    Returns:
        model (tf.keras.Model)
    """

    inputs = tf.keras.Input(shape=input_shape, name='input')
    x = inputs
    x = tf.keras.layers.Conv1D(
            32, 24,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4)
        )(x)
    x = tf.keras.layers.Dropout(0.1)(x)

    x = tf.keras.layers.Conv1D(
            64, 16,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
        )(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    
    x = tf.keras.layers.Conv1D(
        96, 8,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
        )(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    
    if use_standard_max_pooling:
        x = tf.keras.layers.MaxPool1D(pool_size=x.shape[1], padding='valid', data_format='channels_last', name='max_pooling1d')(x)
        x = tf.keras.layers.Reshape([x.shape[-1]], name='reshape_squeeze')(x)
    else:
        x = tf.keras.layers.GlobalMaxPool1D(data_format='channels_last', name='global_max_pooling1d')(x) 

    return tf.keras.Model(inputs, x, name=model_name)


def extract_core_model(composite_model):
    return composite_model.layers[1]

def extract_har_model(multitask_model, optimizer, output_index=-1, model_name="har"):
    model = tf.keras.Model(inputs=multitask_model.inputs, outputs=multitask_model.outputs[output_index], name=model_name)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy"), tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
    )

    return model

def set_freeze_layers(model, num_freeze_layer_index=None):
    if num_freeze_layer_index is None:
        for layer in model.layers:
            layer.trainable = False
    else:
        for layer in model.layers[:num_freeze_layer_index]:
            layer.trainable = False
        for layer in model.layers[num_freeze_layer_index:]:
            layer.trainable = True


def attach_full_har_classification_head(core_model, output_shape, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), num_units=1024, model_name="HAR"):
    """
    Create a full 2-layer classification model from the base mode, using activitations from an intermediate layer with partial freezing
    Architecture:
        base_model-intermediate_layer
        -> Dense: 1024 units
        -> ReLU
        -> Dense: output_shape units
        -> Softmax
    
    Optimizer: Adam
    Loss: CategoricalCrossentropy
    Parameters:
        base_model
            the base model from which the activations are extracted
        
        output_shape
            number of output classifiction categories
        model_name
            name of the output model
        intermediate_layer
            the index of the intermediate layer from which the activations are extracted
        last_freeze_layer
            the index of the last layer to be frozen for fine-tuning (including the layer with the index)
    
    Returns:
        trainable_model (tf.keras.Model)
    """

    inputs = tf.keras.Input(shape=core_model.input.shape[1:], name='input')
    intermediate_x = core_model(inputs)

    x = tf.keras.layers.Dense(num_units, activation='relu')(intermediate_x)
    x = tf.keras.layers.Dense(output_shape)(x)
    outputs = tf.keras.layers.Softmax()(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy"), tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
    )

    return model


def attach_linear_classification_head(core_model, output_shape, optimizer=tf.keras.optimizers.SGD(learning_rate=0.03), model_name="Linear"):

    """
    Create a linear classification model from the base mode, using activitations from an intermediate layer
    Architecture:
        base_model-intermediate_layer
        -> Dense: output_shape units
        -> Softmax
    
    Optimizer: SGD
    Loss: CategoricalCrossentropy
    Parameters:
        base_model
            the base model from which the activations are extracted
        
        output_shape
            number of output classifiction categories
        intermediate_layer
            the index of the intermediate layer from which the activations are extracted
    
    Returns:
        trainable_model (tf.keras.Model)
    """

    inputs = tf.keras.Input(shape=core_model.input.shape[1:], name='input')
    intermediate_x = core_model(inputs)

    x = tf.keras.layers.Dense(output_shape, kernel_initializer=tf.random_normal_initializer(stddev=.01))(intermediate_x)
    outputs = tf.keras.layers.Softmax()(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy"), tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
    )
    return model


def attach_multitask_transform_head(core_model, output_tasks, optimizer, with_har_head=False, har_output_shape=None, num_units_har=1024, model_name="multitask_transform"):
    """
    Note: core_model is also modified after training this model (i.e. the weights are updated)
    """
    inputs = tf.keras.Input(shape=core_model.input.shape[1:], name='input')
    intermediate_x = core_model(inputs)
    outputs = []
    losses = [tf.keras.losses.BinaryCrossentropy() for _ in output_tasks]
    for task in output_tasks:
        x = tf.keras.layers.Dense(256, activation='relu')(intermediate_x)
        pred = tf.keras.layers.Dense(1, activation='sigmoid', name=task)(x)
        outputs.append(pred)


    if with_har_head:
        x = tf.keras.layers.Dense(num_units_har, activation='relu')(intermediate_x)
        x = tf.keras.layers.Dense(har_output_shape)(x)
        har_pred = tf.keras.layers.Softmax(name='har')(x)

        outputs.append(har_pred)
        losses.append(tf.keras.losses.CategoricalCrossentropy())

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)

    model.compile(
        optimizer=optimizer,
        loss=losses,
        metrics=['accuracy']
    )
    
    return model


def save_model_results(model, x_test, y_test, filename):
        # Save model summary to file
    with open(filename, 'a') as file:
        file.write("\n\n=== Model Summary ===\n")
        model.summary(print_fn=lambda x: file.write(x + '\n'))

    # Evaluate the model
    test_result = model.evaluate(x_test, y_test, return_dict=True)

    # Save test results to file
    with open(filename, 'a') as file:
        file.write(f"\nTest Result: {test_result}\n")

    # Generate predictions
    predicted_labels = np.argmax(model.predict(x_test), axis=1)
    true_labels = np.argmax(y_test, axis=1)

    # Compute confusion matrix
    cm = tf.math.confusion_matrix(true_labels, predicted_labels)

    # Save confusion matrix to file
    with open(filename, 'a') as file:
        file.write("\nConfusion Matrix:\n")
        np.savetxt(file, cm, fmt='%d')


def create_CNN_LSTM_Model(input_shape, model_name="CNN_LSTM"):
    """
    Combine the CNN model proposed in SelfHAR with a bidirectional LSTM model
    The model is inspired by the paper below
    @INPROCEEDINGS{9065078,
    author={Mutegeki, Ronald and Han, Dong Seog},
    booktitle={2020 International Conference on Artificial Intelligence in Information and Communication (ICAIIC)}, 
    title={A CNN-LSTM Approach to Human Activity Recognition}, 
    year={2020},
    volume={},
    number={},
    pages={362-366},
    doi={10.1109/ICAIIC48513.2020.9065078}
    }
    
    Parameters:
        input_shape
            the input shape for the model, should be (window_size, num_channels)
    
    Returns:
        model (tf.keras.Model)
    """
    inputs = tf.keras.Input(shape=input_shape, name='input')
    x = inputs
    x = tf.keras.layers.Dense(300, activation='relu')(x)
    x = tf.keras.layers.Conv1D(
            64, 3,
            activation='relu',
            strides = 1,
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4)
        )(x)
    x = tf.keras.layers.Conv1D(
            128, 3,
            activation='relu',
            strides = 1,
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
        )(x)
    x = tf.keras.layers.MaxPool1D(pool_size=4, padding='valid', data_format='channels_last', strides=2)(x)

    x = tf.keras.layers.Conv1D(
        32, 5,
        activation='relu',
        strides = 2,
        kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
        )(x)
    x = tf.keras.layers.Conv1D(
        32, 5,
        activation='relu',
        strides = 2,
        kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
        )(x)
    x = tf.keras.layers.MaxPool1D(pool_size=4, padding='valid', data_format='channels_last', strides=2)(x)

    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(96))(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    # x = tf.keras.layers.Dense(96, activation='relu')(x)
    outputs = tf.keras.layers.Dense(100, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs, name=model_name)


def create_LSTM_Model(input_shape, model_name="LSTM"):
    """
        title={Deep residual bidir-LSTM for human activity recognition using wearable sensors},
        author={Zhao, Yu and Yang, Rennong and Chevalier, Guillaume and Xu, Ximeng and Zhang, Zhenxing},
        journal={Mathematical Problems in Engineering},
        volume={2018},
        pages={1--13},
        year={2018},
        publisher={Hindawi Limited}
    """
    inputs = tf.keras.Input(shape=input_shape, name='input')
    x = inputs    
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(96))(x)
    x = tf.keras.layers.Dense(96, activation='relu')(x)
    outputs = tf.keras.layers.Dense(100, activation='sigmoid')(x)

    return tf.keras.Model(inputs, outputs, name= model_name)


def create_LSTM_CNN_Model(input_shape, model_name="LSTM_CNN"):
    """
    This model is inspired by Xia, Huang and Wang's LSTM-CNN model from the paper "LSTM-CNN Architecture for Human Activity Recognition" 

    @ARTICLE{9043535,
    author={Xia, Kun and Huang, Jianguang and Wang, Hanyu},
    journal={IEEE Access}, 
    title={LSTM-CNN Architecture for Human Activity Recognition}, 
    year={2020},
    volume={8},
    number={},
    pages={56855-56866},
    doi={10.1109/ACCESS.2020.2982225}}

    Architecture:
        Input
        -> LSTM
        -> LSTM
        -> Conv 1D
        -> Max Pooling
        -> Conv 1D
        -> global max pooling


    Parameters:
        input_shape
            the input shape for the model, should be (window_size, num_channels)
    
    Returns:
        model (tf.keras.Model)
    """
    inputs = tf.keras.Input(shape=(400,3), name='input')
    x = tf.keras.layers.LSTM(32, return_sequences=True)(inputs)
    x = tf.keras.layers.LSTM(32, return_sequences=True)(x)
    x = tf.keras.layers.Conv1D(
            64, 5,
            activation='relu',
            strides = 2,
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4)
        )(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2, padding='valid', data_format='channels_last', strides=2)(x)

    x = tf.keras.layers.Conv1D(
            128, 3,
            activation='relu',
            strides =1,
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4)
        )(x)

    outputs = tf.keras.layers.GlobalMaxPool1D(data_format='channels_last', name='global_max_pooling1d')(x) 

    model = tf.keras.Model(inputs, outputs)
    return model


def create_transformer_model(input_shape, model_name="Transformer"):
    """
    @misc{kearstransformer, title={Keras Documentation: Timeseries classification with a Transformer model}, 
    url={https://keras.io/examples/timeseries/timeseries_classification_transformer/}, author={Ntakouris, Theodoros}, year={2021}, month={Jun}} 
    """
    inputs = tf.keras.Input(shape=input_shape, name='input')
    x = inputs
    x = tf.keras.layers.MultiHeadAttention(
        key_dim=512,
        num_heads=16,
        dropout=0.25,
    )(x, x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    x = tf.keras.layers.Dense(128 , activation='relu')(res)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.Dense(3)(x)
    x = x + res
    x = tf.keras.layers.GlobalMaxPool1D(data_format='channels_last', name='global_max_pooling1d')(x) 

    outputs = tf.keras.layers.Softmax()(x)
    return tf.keras.Model(inputs, outputs, name=model_name)