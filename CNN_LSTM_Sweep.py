import tensorflow as tf

from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
import pickle
import data_pre_processing
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb
from wandb.keras import WandbCallback
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

import self_har_models
tf.get_logger().setLevel('INFO')


wandb.login(key='d75a632eee4f2a1869dfe5344e9e2299aca50ebd')
# get data from pickle 
with open('pickled_datasets/pamap.pickle', 'rb') as file:
    pamap_df = pickle.load(file)
with open('pickled_datasets/hhar.pickle', 'rb') as file:
    hhar_df = pickle.load(file)
with open('pickled_datasets/motionsense.pickle', 'rb') as file:
    motion_sense_df = pickle.load(file)
    
def concat_datasets(datasets, sensor_type):
    concated_datasets = {}
    ##TODO need to check key to see whether mag exists
    for df in datasets:
        concated_datasets.update(df[sensor_type])
    return concated_datasets

cdf = concat_datasets([pamap_df, hhar_df, motion_sense_df], "acc")
def get_labels(data):
    all_labels = []
    for user in data:
        all_labels = np.concatenate((np.unique(data[user][0][1]), all_labels))
    labels = np.unique(all_labels)
    return labels

labels = get_labels(cdf)
label_map = {label: index for index, label in enumerate(labels)}
print(label_map)
user_datasets_processed = data_pre_processing.pre_process_dataset_composite(
    cdf, label_map, 6, 
    ['101', '102', '103', '104', '105', '106', '107', 'a', 'b', 'c', 'd', 'e', 'f', 'g', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20'], 
    ["108", "109", "h", "i", "21", "22", "23", "24"], 400, 200
)

# implement early stopping
callback = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

sweep_config = {
  'method': 'grid', 
  'metric': {
      'name': 'epoch/val_loss',
      'goal': 'minimize'
  },
  'early_terminate':{
      'type': 'hyperband',
      'min_iter': 5
  },
  'parameters': {
      'batch_size': {
          'values': [32, 64, 128, 256]
      },
      'dropout1': {
          'values': [0.3, 0.4, 0.5]
        },
      'dropout2': {
          'values': [0.3, 0.4, 0.5]
        },
      'layer1': {
          'values': [32, 64, 128, 256, 512]
      },
      'layer2': {
          'values': [32, 64, 128, 256, 512]
      },
      'dense1': {
          'values': [50, 100, 150, 200, 250, 300]
      },
      'dense2': {
          'values': [50, 100, 150, 200, 250]
      }, 
      'layer3': {
          'values': [32, 64, 128, 256, 512]
      },
      'layer4': {
          'values': [32, 64, 128, 256, 512]
      }
  }
}

sweep_id = wandb.sweep(sweep_config, project="sweep4")

def sweep_train(config_defaults=None):
    wandb.init(project="sweep4")
    # an initial value
    configs = {
        'layer1': 128,
        'layer2': 64,
        'dropout1': 0.3,
        'dropout2': 0.3,
        'dense1': 50,
        'dense2': 50,
        'layer3': 32,
        'layer4': 32
    }

    # Specify the other hyperparameters to the configuration
    config = wandb.config
    config.epochs = 5



    inputs = tf.keras.Input(shape=(400,3), name='input')
    x = inputs
    x = tf.keras.layers.Dense(wandb.config.dense1, activation='relu')(x)
    x = tf.keras.layers.Conv1D(
            wandb.config.layer1, 3,
            activation='relu',
            strides = 1,
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4)
        )(x)
    x = tf.keras.layers.Conv1D(
            wandb.config.layer2, 3,
            activation='relu',
            strides = 1,
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
        )(x)
    x = tf.keras.layers.MaxPool1D(pool_size=4, padding='valid', data_format='channels_last', strides=2)(x)

    x = tf.keras.layers.Conv1D(
        wandb.config.layer3, 5,
        activation='relu',
        strides = 2,
        kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
        )(x)
    x = tf.keras.layers.Conv1D(
        wandb.config.layer4, 5,
        activation='relu',
        strides = 2,
        kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
        )(x)
    x = tf.keras.layers.MaxPool1D(pool_size=4, padding='valid', data_format='channels_last', strides=2)(x)

    x = tf.keras.layers.Dropout(wandb.config.dropout1)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(96))(x)
    x = tf.keras.layers.Dropout(wandb.config.dropout2)(x)
    # x = tf.keras.layers.Dense(96, activation='relu')(x)
    x = tf.keras.layers.Dense(wandb.config.dense2, activation='softmax')(x)

    CNNLSTN_model = tf.keras.Model(inputs, x, name="CNN-LSTM")
    full_CNNLSTM_model = self_har_models.attach_full_har_classification_head(CNNLSTN_model, 6, optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001))


    wandb.config.architecture_name = "CNN-LSTM"


    # Prepare the metrics.
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    
    full_CNNLSTM_model.fit(user_datasets_processed[0][0], user_datasets_processed[0][1]
                    , epochs=100, validation_data=(user_datasets_processed[1][0], user_datasets_processed[1][1]), callbacks=[callback, WandbMetricsLogger(log_freq=5),
                      WandbModelCheckpoint("models")])
    
wandb.agent(sweep_id, function=sweep_train, count=1000)