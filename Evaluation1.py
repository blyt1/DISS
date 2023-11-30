import self_har_models
import dataset_pre_processing
import raw_data_processing
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def prepare_self_supervised_learning_dataset(sensor_type, train_users, test_users):
    with open('pickled_datasets/pamap2.pickle', 'rb') as file:
        pamap_df = pickle.load(file)
    with open('pickled_datasets/hhar2.pickle', 'rb') as file:
        hhar_df = pickle.load(file)
    with open('pickled_datasets/motionsense2.pickle', 'rb') as file:
        motion_sense_df = pickle.load(file)

    cdf = dataset_pre_processing.concat_datasets([pamap_df, hhar_df, motion_sense_df], sensor_type=sensor_type)
    labels = dataset_pre_processing.get_labels(cdf)
    label_map = {label: index for index, label in enumerate(labels)}
    user_dataset_preprocessed = dataset_pre_processing.pre_process_dataset_composite(
        user_datasets=cdf, 
        label_map=label_map, 
        output_shape=14,
        train_users=train_users,
        test_users=test_users,
        window_size=400, 
        shift=200
    )
    return user_dataset_preprocessed, labels


def train_self_supervised_model(df, core_model, label_size, optimizer):
    composite_model = self_har_models.attach_full_har_classification_head(core_model=core_model, 
                                                                          output_shape=label_size, 
                                                                          optimizer=optimizer)
    history = composite_model.fit(df[0][0], df[0][1]
                    , epochs=100, validation_data=(df[1][0], df[1][1]))
    return history, composite_model


def eval_model(df, labels, model):
    cnn_test_result = model.evaluate(df[2][0],  df[2][1], return_dict=True)
    print(cnn_test_result)
    predicted_labels = np.argmax(model.predict(df[2][0]), axis=1)
    true_labels = np.argmax(df[2][1], axis=1)
    confusion_mat = tf.math.confusion_matrix(true_labels, predicted_labels)
    cm = confusion_mat.numpy()
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.plot()
    plt.show()

def downstream_testing(df, model, label_size, optimizer):
    core_model = self_har_models.extract_core_model(model)
    har_model = self_har_models.attach_full_har_classification_head(core_model=core_model, 
                                                                          output_shape=label_size, 
                                                                          optimizer=optimizer)
    history = har_model.fit(df[0][0], df[0][1]
                    , epochs=100, validation_data=(df[1][0], df[1][1]))
    return history, har_model


def eval_pamap():
    with open('pickled_datasets/pamap2.pickle', 'rb') as file:
        pamap_df = pickle.load(file)
    pamap_df = dataset_pre_processing.concat_datasets([pamap_df], sensor_type='all')

    labels = dataset_pre_processing.get_labels(pamap_df)
    label_map = {label: index for index, label in enumerate(labels)}
    user_dataset_preprocessed = dataset_pre_processing.pre_process_dataset_composite(
        user_datasets=pamap_df, 
        label_map=label_map, 
        output_shape=9,
        train_users=['101', '102', '103', '104', '105', '106', '107'],
        test_users=["108", "109"],
        window_size=400, 
        shift=200
    )
    core_model = self_har_models.create_CNN_LSTM_Model((400,3))
    history, composite_model = train_self_supervised_model(user_dataset_preprocessed, core_model, 9, tf.keras.optimizers.Adam(learning_rate=0.0005))

    eval_model(user_dataset_preprocessed, labels, composite_model)

    with open('pickled_datasets/pamap_har.pickle', 'rb') as file:
        pamap_har_df = pickle.load(file)

    pamap_har_df = dataset_pre_processing.concat_datasets([pamap_har_df], sensor_type='all')
    har_labels = dataset_pre_processing.get_labels(pamap_har_df)

    har_label_map = {label: index for index, label in enumerate(har_labels)}
    har_preprocessed = dataset_pre_processing.pre_process_dataset_composite(
        user_datasets=pamap_har_df,
        label_map=har_label_map,
        output_shape=19,
        train_users=['101', '102', '103', '104', '105', '106'],
        test_users=['107'],
        window_size=400,
        shift=200
    )

    ds_history, har_model = downstream_testing(har_preprocessed, composite_model, 19, tf.keras.optimizers.Adam(learning_rate=0.0005))
    eval_model(har_preprocessed, labels, har_model)

if __name__ == '__main__':
    eval_pamap()
    pass