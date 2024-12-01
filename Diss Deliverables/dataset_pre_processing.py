import numpy as np
import pandas as pd
import scipy.stats
import sklearn
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
__author__ = "C. I. Tang"
__copyright__ = "Copyright (C) 2021 C. I. Tang"

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
"""
def get_labels(data):
    all_labels = []
    for user in data:
        all_labels = np.concatenate((np.unique(data[user][0][1]), all_labels))
    labels = np.unique(all_labels)
    return labels

def concat_datasets(datasets, sensor_type):
    concated_datasets = {}
    ##TODO need to check key to see whether mag exists
    for df in datasets:
        concated_datasets.update(df[sensor_type])
    return concated_datasets

def get_mode(np_array):
    """
    Get the mode (majority/most frequent value) from a 1D array
    """
    unique_values, counts = np.unique(np_array, return_counts=True)
    mode_index = np.argmax(counts)
    mode_value = unique_values[mode_index]
    return mode_value

def sliding_window_np(X, window_size, shift, stride, offset=0, flatten=None):
    """
    Create sliding windows from an ndarray
    Parameters:
    
        X (numpy-array)
            The numpy array to be windowed
        
        shift (int)
            number of timestamps to shift for each window
            (200 here refers to 50% overlap, no overlap if =400)
        stride (int)
            stride of the window (dilation)
        offset (int)
            starting index of the first window
        
        flatten (function (array) -> (value or array) )
            the function to be applied to a window after it is extracted
            can be used with get_mode (see above) for extracting the label by majority voting
            ignored if is None
    Return:
        Windowed ndarray
            shape[0] is the number of windows
    """

    overall_window_size = (window_size - 1) * stride + 1
    num_windows = (X.shape[0] - offset - (overall_window_size)) // shift + 1
    windows = []
    for i in range(num_windows):
        start_index = i * shift + offset
        this_window = X[start_index : start_index + overall_window_size : stride]
        if flatten is not None:
            this_window = flatten(this_window)
        windows.append(this_window)
    return np.array(windows)


def get_windows_dataset_from_user_list_format(user_datasets, window_size=400, shift=200, stride=1, verbose=0):
    """
    Create windows dataset in 'user-list' format using sliding windows
    Parameters:
        user_datasets
            dataset in the 'user-list' format {user_id: [(sensor_values, activity_labels)]}
        
        window_size = 400
            size of the window (output)
        shift = 200
            number of timestamps to shift for each window
            (200 here refers to 50% overlap, no overlap if =400)
        stride = 1
            stride of the window (dilation)
        verbose = 0
            debug messages are printed if > 0
    
    Return:
        user_dataset_windowed
            Windowed version of the user_datasets
            Windows from different trials are combined into one array
            type: {user_id: ( windowed_sensor_values, windowed_activity_labels)}
            windowed_sensor_values have shape (num_window, window_size, channels)
            windowed_activity_labels have shape (num_window)
            Labels are decided by majority vote
    """
    
    user_dataset_windowed = {}

    for user_id in user_datasets:
        if verbose > 0:
            print(f"Processing {user_id}")
        x = []
        y = []

        # Loop through each trail of each user
        for v,l in user_datasets[user_id]:
            v_windowed = sliding_window_np(v, window_size, shift, stride)
            
            # flatten the window by majority vote (1 value for each window)
            l_flattened = sliding_window_np(l, window_size, shift, stride, flatten=get_mode)
            if len(v_windowed) > 0:
                x.append(v_windowed)
                y.append(l_flattened)
            if verbose > 0:
                print(f"Data: {v_windowed.shape}, Labels: {l_flattened.shape}")

        # combine all trials
        user_dataset_windowed[user_id] = (np.concatenate(x), np.concatenate(y).squeeze())
    return user_dataset_windowed


def combine_windowed_dataset(user_datasets_windowed, train_users, test_users=None, verbose=0):
    """
    Combine a windowed 'user-list' dataset into training and test sets
    Parameters:
        user_dataset_windowed
            dataset in the windowed 'user-list' format {user_id: ( windowed_sensor_values, windowed_activity_labels)}
        
        train_users
            list or set of users (corresponding to the user_id) to be used as training data
        test_users = None
            list or set of users (corresponding to the user_id) to be used as testing data
            if is None, then all users not in train_users will be treated as test users 
        verbose = 0
            debug messages are printed if > 0
    Return:
        (train_x, train_y, test_x, test_y)
            train_x, train_y
                the resulting training/test input values as a single numpy array
            test_x, test_y
                the resulting training/test labels as a single (1D) numpy array
    """
    
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for user_id in user_datasets_windowed:
        
        v,l = user_datasets_windowed[user_id]
        if user_id in train_users:
            if verbose > 0:
                print(f"{user_id} Train")
            train_x.append(v)
            train_y.append(l)
        elif test_users is None or user_id in test_users:
            if verbose > 0:
                print(f"{user_id} Test")
            test_x.append(v)
            test_y.append(l)
    

    if len(train_x) == 0:
        train_x = np.array([])
        train_y = np.array([])
    else:
        train_x = np.concatenate(train_x)
        train_y = np.concatenate(train_y).squeeze()
    
    if len(test_x) == 0:
        test_x = np.array([])
        test_y = np.array([])
    else:
        test_x = np.concatenate(test_x)
        test_y = np.concatenate(test_y).squeeze()

    return train_x, train_y, test_x, test_y


def apply_label_map(y, label_map):
    """
    Apply a dictionary mapping to an array of labels
    Can be used to convert str labels to int labels
    Parameters:
        y
            1D array of labels
        label_map
            a label dictionary of (label_original -> label_new)
    Return:
        y_mapped
            1D array of mapped labels
            None values are present if there is no entry in the dictionary
    """

    y_mapped = []
    for l in y:
        y_mapped.append(label_map.get(l))
    return np.array(y_mapped)


def filter_none_label(X, y):
    """
    Filter samples of the value None
    Can be used to exclude non-mapped values from apply_label_map
    Parameters:
        X
            data values
        y
            labels (1D)
    Return:
        (X_filtered, y_filtered)
            X_filtered
                filtered data values
            
            y_filtered
                filtered labels (of type int)
    """

    valid_mask = np.where(y != None)
    return (np.array(X[valid_mask]), np.array(y[valid_mask], dtype=int))

def get_mean_std_from_user_list_format(user_datasets, train_users):
    """
    Obtain and means and standard deviations from a 'user-list' dataset (channel-wise)
    from training users only
    Parameters:
        user_datasets
            dataset in the 'user-list' format {user_id: [(sensor_values, activity_labels)]}
        
        train_users
            list or set of users (corresponding to the user_ids) from which the mean and std are extracted
    Return:
        (means, stds)
            means and stds of the particular users (channel-wise)
            shape: (num_channels)
    """
    
    mean_std_data = []
    for u in train_users:
        for data, _ in user_datasets[u]:
            mean_std_data.append(data)
    mean_std_data_combined = np.concatenate(mean_std_data)
    means = np.mean(mean_std_data_combined, axis=0)
    stds = np.std(mean_std_data_combined, axis=0)
    return (means, stds)

def normalise(data, mean, std):
    """
    Normalise data (Z-normalisation)
    """

    return ((data - mean) / std)

def pre_process_dataset_composite(user_datasets, label_map, output_shape, train_users, test_users, window_size, shift, normalise_dataset=True, validation_split_proportion=0.2, verbose=0):
    """
    A composite function to process a dataset
    Steps
        1: Use sliding window to make a windowed dataset (see get_windows_dataset_from_user_list_format)
        2: Split the dataset into training and test set (see combine_windowed_dataset)
        3: Normalise the datasets (see get_mean_std_from_user_list_format)
        4: Apply the label map and filter labels (see apply_label_map, filter_none_label)
        5: One-hot encode the labels (see tf.keras.utils.to_categorical)
        6: Split the training set into training and validation sets (see sklearn.model_selection.train_test_split)
    
    Parameters:
        user_datasets
            dataset in the 'user-list' format {user_id: [(sensor_values, activity_labels)]}
        label_map
            a mapping of the labels
            can be used to filter labels
            (see apply_label_map and filter_none_label)
        output_shape
            number of output classifiction categories
            used in one hot encoding of the labels
            (see tf.keras.utils.to_categorical)
        train_users
            list or set of users (corresponding to the user_id) to be used as training data
        test_users
            list or set of users (corresponding to the user_id) to be used as testing data
        window_size
            size of the data windows
            (see get_windows_dataset_from_user_list_format)
        shift
            number of timestamps to shift for each window
            (see get_windows_dataset_from_user_list_format)
        normalise_dataset = True
            applies Z-normalisation if True
        validation_split_proportion = 0.2
            if not None, the proportion for splitting the full training set further into training and validation set using random sampling
            (see sklearn.model_selection.train_test_split)
            if is None, the training set will not be split - the return value np_val will also be none
        verbose = 0
            debug messages are printed if > 0
    
    Return:
        (np_train, np_val, np_test)
            three pairs of (X, y)
            X is a windowed set of data points
            y is an array of one-hot encoded labels
            if validation_split_proportion is None, np_val is None
    """
    print("here1")
    # Step 1
    user_datasets_windowed = get_windows_dataset_from_user_list_format(user_datasets, window_size=window_size, shift=shift)
    print("splitting training")
    # Step 2
    train_x, train_y, test_x, test_y = combine_windowed_dataset(user_datasets_windowed, train_users, test_users)
    print("normalising")
    # Step 3
    if normalise_dataset:
        means, stds = get_mean_std_from_user_list_format(user_datasets, train_users)
        train_x = normalise(train_x, means, stds)
        if len(test_users) > 0:
            test_x = normalise(test_x, means, stds)
        else:
            test_x = np.empty((0,0))
    print("mapping")
    # Step 4
    train_y_mapped = apply_label_map(train_y, label_map)
    test_y_mapped = apply_label_map(test_y, label_map)

    train_x, train_y_mapped = filter_none_label(train_x, train_y_mapped)
    test_x, test_y_mapped = filter_none_label(test_x, test_y_mapped)

    if verbose > 0:
        print("Test")
        print(np.unique(test_y, return_counts=True))
        print(np.unique(test_y_mapped, return_counts=True))
        print("-----------------")

        print("Train")
        print(np.unique(train_y, return_counts=True))
        print(np.unique(train_y_mapped, return_counts=True))
        print("-----------------")

    # Step 5
    train_y_one_hot = tf.keras.utils.to_categorical(train_y_mapped, num_classes=output_shape)
    test_y_one_hot = tf.keras.utils.to_categorical(test_y_mapped, num_classes=output_shape)

    r = np.random.randint(len(train_y_mapped))
    assert train_y_one_hot[r].argmax() == train_y_mapped[r]
    if len(test_users) > 0:
        r = np.random.randint(len(test_y_mapped))
        assert test_y_one_hot[r].argmax() == test_y_mapped[r]

    # Step 6
    if validation_split_proportion is not None and validation_split_proportion > 0:
        train_x_split, val_x_split, train_y_split, val_y_split = sklearn.model_selection.train_test_split(train_x, train_y_one_hot, test_size=validation_split_proportion, random_state=42)
    else:
        train_x_split = train_x
        train_y_split = train_y_one_hot
        val_x_split = None
        val_y_split = None
        

    if verbose > 0:
        print("Training data shape:", train_x_split.shape)
        print("Validation data shape:", val_x_split.shape if val_x_split is not None else "None")
        print("Testing data shape:", test_x.shape)

    np_train = (train_x_split, train_y_split)
    np_val = (val_x_split, val_y_split) if val_x_split is not None else None
    np_test = (test_x, test_y_one_hot)

    # original_np_train = np_train
    # original_np_val = np_val
    # original_np_test = np_test

    return (np_train, np_val, np_test)

### this probably doesn't work ngl 
def composite_pre_processing(dataset, output_shape=3, train_device="acc", test_device=None, window_size=400,
                             shift=200, normalise_dataset=True, validation_split_proportion=0.2, verbose=0):
    windowed_dataset = get_windows_dataset(dataset, window_size=window_size, shift=shift)
    df = windowed_dataset['acc']
    v, l = df
    print(v)
    train_x, test_x, train_y, test_y = train_test_split(v, l, test_size=0.2, stratify=l, random_state=42)
    if normalise_dataset:
        normalise(train_x)
    # get label maps
    labels = dataset['acc'][0][1]
    one_hot_mapping = create_one_hot_mapping(labels)
    train_y_mapped = one_hot_mapping.transform(train_y)
    test_y_mapped = one_hot_mapping.transform(test_y)

    train_y_one_hot = tf.keras.utils.to_categorical(train_y_mapped, num_classes=output_shape)
    test_y_one_hot = tf.keras.utils.to_categorical(test_y_mapped, num_classes=output_shape)
    # train_y_one_hot = train_y_mapped
    # test_y_one_hot = test_y_mapped
    # r = np.random.randint(len(train_y_mapped))
    # assert np.array_equal(train_y_one_hot[r].argmax(), train_y_mapped[r])

    # if len(test_device) > 0:
    #     r = np.random.randint(len(test_y_mapped))
    #     assert test_y_one_hot[r].argmax() == test_y_mapped[r]
    # Step 6
    if validation_split_proportion is not None and validation_split_proportion > 0:
        train_x_split, val_x_split, train_y_split, val_y_split = sklearn.model_selection.train_test_split(train_x,
                                                                                                          train_y_one_hot,
                                                                                                          test_size=validation_split_proportion,
                                                                                                          random_state=42)
    else:
        train_x_split = train_x
        train_y_split = train_y_one_hot
        val_x_split = None
        val_y_split = None

    if verbose > 0:
        print("Training data shape:", train_x_split.shape)
        print(train_y_split.shape)
        print("Validation data shape:", val_x_split.shape if val_x_split is not None else "None")
        print("Testing data shape:", test_x.shape)
    print(train_y_split)
    np_train = (train_x_split, train_y_split)
    np_val = (val_x_split, val_y_split) if val_x_split is not None else None
    np_test = (test_x, test_y_one_hot)

    # original_np_train = np_train
    # original_np_val = np_val
    # original_np_test = np_test

    return (np_train, np_val, np_test)
