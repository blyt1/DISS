import glob
import re
import os
import pandas as pd
import numpy as np
import pickle

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


def process_motion_sense_accelerometer_files(accelerometer_data_folder_path):
    """
    Preprocess the accelerometer files of the MotionSense dataset into the 'user-list' format
    Data files can be found at https://github.com/mmalekzadeh/motion-sense/tree/master/data
    Parameters:
        accelerometer_data_folder_path (str):
            the path to the folder containing the data files (unzipped)
            e.g. motionSense/B_Accelerometer_data/
            the trial folders should be directly inside it (e.g. motionSense/B_Accelerometer_data/dws_1/)
    Return:
        
        user_datsets (dict of {user_id: [(sensor_values, activity_labels)]})
            the processed dataset in a dictionary, of type {user_id: [(sensor_values, activity_labels)]}
            the keys of the dictionary is the user_id (participant id)
            the values of the dictionary are lists of (sensor_values, activity_labels) pairs
                sensor_values are 2D numpy array of shape (length, channels=3)
                activity_labels are 1D numpy array of shape (length)
                each pair corresponds to a separate trial 
                    (i.e. time is not contiguous between pairs, which is useful for making sliding windows, where it is easy to separate trials)
    """

    # label_set = {}
    user_datasets = {}
    all_trials_folders = sorted(glob.glob(accelerometer_data_folder_path + "/*"))

    # Loop through every trial folder
    for trial_folder in all_trials_folders:
        trial_name = os.path.split(trial_folder)[-1]

        # label of the trial is given in the folder name, separated by underscore
        label = trial_name.split("_")[0]
        # label_set[label] = True
        print(trial_folder)

        # Loop through files for every user of the trail
        for trial_user_file in sorted(glob.glob(trial_folder + "/*.csv")):

            # use regex to match the user id
            user_id_match = re.search(r'(?P<user_id>[0-9]+)\.csv', os.path.split(trial_user_file)[-1])
            if user_id_match is not None:
                user_id = int(user_id_match.group('user_id'))

                # Read file
                user_trial_dataset = pd.read_csv(trial_user_file)
                # user_trial_dataset.dropna(how="any", inplace=True)

                # Extract the x, y, z channels
                values = user_trial_dataset[["x", "y", "z"]].values

                # the label is the same during the entire trial, so it is repeated here to pad to the same length as the values
                labels = np.repeat(label, values.shape[0])

                if user_id not in user_datasets:
                    user_datasets[user_id] = []
                user_datasets[user_id].append((values, labels))
            else:
                print("[ERR] User id not found", trial_user_file)

    return user_datasets


def process_hhar_accelerometer_files(data_folder_path):
    """
    Preprocess the accelerometer files of the HHAR dataset into the 'user-list' format
    Data files can be found at http://archive.ics.uci.edu/ml/datasets/heterogeneity+activity+recognition
    
    """
    # print(data_folder_path)

    har_dataset = pd.read_csv(
        os.path.join(data_folder_path, 'Phones_accelerometer.csv'))  # "<PATH_TO_HHAR_DATASET>/Phones_accelerometer.csv"
    har_dataset.dropna(how="any", inplace=True)
    har_dataset = har_dataset[["x", "y", "z", "gt", "User"]]
    har_dataset.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    har_users = har_dataset["user-id"].unique()

    user_datasets = {}
    for user in har_users:
        user_extract = har_dataset[har_dataset["user-id"] == user]
        data = user_extract[["x-axis", "y-axis", "z-axis"]].values
        labels = user_extract["activity"].values
        print(f"{user} {data.shape}")
        user_datasets[user] = [(data, labels)]

    return user_datasets


def process_hhar_all_files(data_folder_path):
    """
    Preprocess all files of from the HHAR dataset
    Data files can be found at http://archive.ics.uci.edu/ml/datasets/heterogeneity+activity+recognition
    Args:
        data_folder_path: folder path

    Returns:
        user_datasets (a list containing data collected from the acc + gyro from both smartphones and watches)
    """
    har_phone_acc = pd.read_csv(os.path.join(data_folder_path, 'Phones_accelerometer.csv'))
    har_phone_acc['Device'] = "Phone acc"
    har_phone_gyro = pd.read_csv(os.path.join(data_folder_path, 'Phones_gyroscope.csv'))
    har_phone_gyro['Device'] = "Phone gyro"
    har_watch_acc = pd.read_csv(os.path.join(data_folder_path, 'Watch_accelerometer.csv'))
    har_watch_acc['Device'] = "Watch acc"
    har_watch_gyro = pd.read_csv(os.path.join(data_folder_path, 'Watch_gyroscope.csv'))
    har_watch_gyro['Device'] = "Watch gyro"
    acc_data = pd.concat([har_phone_acc, har_watch_acc])
    gyro_data = pd.concat([har_phone_gyro, har_watch_gyro])

    # acc_data.dropna(how="any", inplace=True)
    acc_data = acc_data[["x", "y", "z", "gt", "User", "Device"]]
    acc_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id", "device"]
    har_users = acc_data["user-id"].unique()

    acc_datasets = {}
    for user in har_users:
        user_extract = acc_data[acc_data["user-id"] == user]
        data = user_extract[["x-axis", "y-axis", "z-axis"]].copy()
        # data["data-source"] = "HHAR"
        labels = user_extract["device"].values
        print(f"{user} {data.shape}")
        acc_datasets[user] = [(data, labels)]

    # gyro_data.dropna(how="any", inplace=True)
    gyro_data = gyro_data[["x", "y", "z", "gt", "User", "Device"]]
    gyro_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id", "device"]
    gyro_datasets = {}
    for user in har_users:
        user_extract = gyro_data[gyro_data["user-id"] == user]
        data = user_extract[["x-axis", "y-axis", "z-axis"]].copy()
        # data["data-source"] = "HHAR"
        labels = user_extract["device"].values
        print(f"{user} {data.shape}")
        gyro_datasets[user] = [(data, labels)]

    all_data = {}
    gyro_acc_data = pd.concat([acc_data, gyro_data])
    gyro_acc_data.dropna(how="any", inplace=True)

    for user in har_users:
        user_extract = gyro_acc_data[gyro_acc_data["user-id"] == user]
        data = user_extract[["x-axis", "y-axis", "z-axis"]].copy()
        # data["data-source"] = "HHAR"
        labels = user_extract["device"].values
        print(f"{user} {data.shape}")
        all_data[user] = [(data, labels)]

    user_datasets = {'acc': acc_datasets,
                     'gyro': gyro_datasets, 
                     'all': all_data}
    return user_datasets


def process_PAMAP2_all_data(data_folder_path):
    combined_data = []

    for filename in os.listdir(data_folder_path):
        if filename.endswith('.dat'):
            file_path = os.path.join(data_folder_path, filename)
            df = pd.read_csv(file_path, sep=' ', header=None, na_filter="NaN")
            df['User-ID'] = filename[7:10]
            combined_data.append(df)
    
    user_df = pd.concat(combined_data, ignore_index=True)
    hand_acc_data = user_df.loc[:, [4, 5, 6, 1, 'User-ID']]
    hand_acc_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    hand_acc_data['device'] = "IMU hand acc"

    hand_gyro_data = user_df.loc[:, [10, 11, 12, 1, 'User-ID']]
    hand_gyro_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    hand_gyro_data['device'] = 'IMU hand gyro'

    hand_mag_data = user_df.loc[:, [13, 14, 15, 1, 'User-ID']]
    hand_mag_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    hand_mag_data['device'] = 'IMU hand mag'

    chest_acc_data = user_df.loc[:, [21, 22, 23, 1, 'User-ID']]
    chest_acc_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    chest_acc_data['device'] = "IMU chest acc"

    chest_gyro_data = user_df.loc[:, [27, 28, 29, 1, 'User-ID']]
    chest_gyro_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    chest_gyro_data['device'] = "IMU chest gyro"

    chest_mag_data = user_df.loc[:, [30, 31, 32, 1, 'User-ID']]
    chest_mag_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    chest_mag_data['device'] = "IMU chest mag"

    ankle_acc_data = user_df.loc[:, [38, 39, 40, 1, 'User-ID']]
    ankle_acc_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    ankle_acc_data['device'] = 'IMU ankle acc'

    ankle_gyro_data = user_df.loc[:, [44, 45, 46, 1, 'User-ID']]
    ankle_gyro_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    ankle_gyro_data['device'] = 'IMU ankle gyro'

    ankle_mag_data = user_df.loc[:, [47, 48, 49, 1, 'User-ID']]
    ankle_mag_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    ankle_mag_data['device'] = 'IMU ankle mag'

    PAMAP_users = user_df['User-ID'].unique()   

    pamap_acc_data = pd.concat([hand_acc_data, chest_acc_data, ankle_acc_data])
    pamap_acc_datasets = {}
    for user in PAMAP_users:
        user_extract = pamap_acc_data[pamap_acc_data["user-id"] == user]
        user_extract = user_extract.dropna()
        data = user_extract[["x-axis", "y-axis", "z-axis"]].copy()
        # data["data-source"] = "PAMAP"
        labels = user_extract["device"].values
        print(f"{user} {data.shape}")
        pamap_acc_datasets[user] = [(data, labels)]
    pamap_gyro_data = pd.concat([hand_gyro_data, chest_gyro_data, ankle_gyro_data])
    pamap_gyro_datasets = {}
    for user in PAMAP_users:
        user_extract = pamap_gyro_data[pamap_gyro_data["user-id"] == user]
        user_extract = user_extract.dropna()
        data = user_extract[["x-axis", "y-axis", "z-axis"]].copy()
        # data["data-source"] = "PAMAP"
        labels = user_extract["device"].values
        print(f"{user} {data.shape}")
        pamap_gyro_datasets[user] = [(data, labels)]

    pamap_mag_data = pd.concat([hand_mag_data, chest_mag_data, ankle_mag_data])
    pamap_mag_datasets = {}
    for user in PAMAP_users:
        user_extract = pamap_mag_data[pamap_mag_data["user-id"] == user]
        user_extract = user_extract.dropna()
        data = user_extract[["x-axis", "y-axis", "z-axis"]].copy()
        # data["data-source"] = "PAMAP"
        labels = user_extract["device"].values
        print(f"{user} {data.shape}")
        pamap_mag_datasets[user] = [(data, labels)]

    pamap_all_data = {}
    gyro_acc_mag_data = pd.concat([pamap_acc_data, pamap_gyro_data, pamap_mag_data])
    for user in PAMAP_users:
        user_extract = gyro_acc_mag_data[gyro_acc_mag_data["user-id"] == user]
        user_extract = user_extract.dropna()
        data = user_extract[["x-axis", "y-axis", "z-axis"]].copy()
        # data["data-source"] = "PAMAP"
        labels = user_extract["device"].values
        print(f"{user} {data.shape}")
        pamap_all_data[user] = [(data, labels)]

    user_datasets = {'acc': pamap_acc_datasets,
                     'gyro': pamap_gyro_datasets, 
                     'mag': pamap_mag_datasets, 
                     'all': pamap_all_data}
    return user_datasets


def concat_datasets(datasets, sensor_type):
    concated_datasets = {}
    ##TODO need to check key to see whether mag exists
    for df in datasets:
        concated_datasets.update(df[sensor_type])
    return concated_datasets


def process_motion_sense_all_files(data_folder_path):
    user_datasets = {}
    all_sensor_folders = sorted(glob.glob(data_folder_path + "/*"))
    all_data = {}
    for folder in all_sensor_folders:
        # print(folder)
        sensor_data = {}
        
        all_trials_folders = sorted(glob.glob(folder + "/*"))
        # Loop through every trial folder
        for trial_folder in all_trials_folders:
            trial_name = os.path.split(trial_folder)[-1]

            # label of the trial is given in the folder name, separated by underscore
            label = "iphone " + folder[46:49]
            # label_set[label] = True
            print(label)

            # Loop through files for every user of the trail
            for trial_user_file in sorted(glob.glob(trial_folder + "/*.csv")):

                # use regex to match the user id
                user_id_match = re.search(r'(?P<user_id>[0-9]+)\.csv', os.path.split(trial_user_file)[-1])
                if user_id_match is not None:
                    user_id = str(int(user_id_match.group('user_id')))

                    # Read file
                    user_trial_dataset = pd.read_csv(trial_user_file)
                    # user_trial_dataset.dropna(how="any", inplace=True)

                    # Extract the x, y, z channels
                    values = user_trial_dataset[["x", "y", "z"]].values

                    # the label is the same during the entire trial, so it is repeated here to pad to the same length as the values
                    labels = np.repeat(label, values.shape[0])

                    if user_id not in sensor_data:
                        sensor_data[user_id] = []
                    if user_id not in all_data:
                        all_data[user_id] = []
                    sensor_data[user_id].append((values, labels))
                    all_data[user_id].append((values, labels))
                else:
                    print("[ERR] User id not found", trial_user_file)
            if(folder[46:49] == "acc"):
                user_datasets.update({'acc': sensor_data})
            else:
                user_datasets.update({'gyro': sensor_data})
    user_datasets.update({'all': all_data})
    return user_datasets

def process_pamap2_har_files(data_folder_path):
    combined_data = []

    for filename in os.listdir(data_folder_path):
        if filename.endswith('.dat'):
            file_path = os.path.join(data_folder_path, filename)
            df = pd.read_csv(file_path, sep=' ', header=None, na_filter="NaN")
            df['User-ID'] = filename[7:10]
            combined_data.append(df)
    
    user_df = pd.concat(combined_data, ignore_index=True)
    hand_acc_data = user_df.loc[:, [4, 5, 6, 1, 'User-ID']]
    hand_acc_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    hand_acc_data['device'] = "IMU hand acc"

    hand_gyro_data = user_df.loc[:, [10, 11, 12, 1, 'User-ID']]
    hand_gyro_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    hand_gyro_data['device'] = 'IMU hand gyro'

    hand_mag_data = user_df.loc[:, [13, 14, 15, 1, 'User-ID']]
    hand_mag_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    hand_mag_data['device'] = 'IMU hand mag'

    chest_acc_data = user_df.loc[:, [21, 22, 23, 1, 'User-ID']]
    chest_acc_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    chest_acc_data['device'] = "IMU chest acc"

    chest_gyro_data = user_df.loc[:, [27, 28, 29, 1, 'User-ID']]
    chest_gyro_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    chest_gyro_data['device'] = "IMU chest gyro"

    chest_mag_data = user_df.loc[:, [30, 31, 32, 1, 'User-ID']]
    chest_mag_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    chest_mag_data['device'] = "IMU chest mag"

    ankle_acc_data = user_df.loc[:, [38, 39, 40, 1, 'User-ID']]
    ankle_acc_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    ankle_acc_data['device'] = 'IMU ankle acc'

    ankle_gyro_data = user_df.loc[:, [44, 45, 46, 1, 'User-ID']]
    ankle_gyro_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    ankle_gyro_data['device'] = 'IMU ankle gyro'

    ankle_mag_data = user_df.loc[:, [47, 48, 49, 1, 'User-ID']]
    ankle_mag_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    ankle_mag_data['device'] = 'IMU ankle mag'
    
    PAMAP_users = user_df['User-ID'].unique()   

    pamap_acc_data = pd.concat([hand_acc_data, chest_acc_data, ankle_acc_data])
    pamap_acc_datasets = {}
    for user in PAMAP_users:
        user_extract = pamap_acc_data[pamap_acc_data["user-id"] == user]
        user_extract = user_extract.dropna()
        data = user_extract[["x-axis", "y-axis", "z-axis"]].copy()
        # data["data-source"] = "PAMAP"
        labels = user_extract["activity"]
        print(f"{user} {data.shape}")
        pamap_acc_datasets[user] = [(data, labels)]
    pamap_gyro_data = pd.concat([hand_gyro_data, chest_gyro_data, ankle_gyro_data])
    pamap_gyro_datasets = {}
    for user in PAMAP_users:
        user_extract = pamap_gyro_data[pamap_gyro_data["user-id"] == user]
        user_extract = user_extract.dropna()
        data = user_extract[["x-axis", "y-axis", "z-axis"]].copy()
        # data["data-source"] = "PAMAP"
        labels = user_extract["activity"]
        print(f"{user} {data.shape}")
        pamap_gyro_datasets[user] = [(data, labels)]

    pamap_mag_data = pd.concat([hand_mag_data, chest_mag_data, ankle_mag_data])
    pamap_mag_datasets = {}
    for user in PAMAP_users:
        user_extract = pamap_mag_data[pamap_mag_data["user-id"] == user]
        user_extract = user_extract.dropna()
        data = user_extract[["x-axis", "y-axis", "z-axis"]].copy()
        # data["data-source"] = "PAMAP"
        labels = user_extract["activity"]
        print(f"{user} {data.shape}")
        pamap_mag_datasets[user] = [(data, labels)]

    pamap_all_data = {}
    gyro_acc_mag_data = pd.concat([pamap_acc_data, pamap_gyro_data, pamap_mag_data])
    for user in PAMAP_users:
        user_extract = gyro_acc_mag_data[gyro_acc_mag_data["user-id"] == user]
        user_extract = user_extract.dropna()
        data = user_extract[["x-axis", "y-axis", "z-axis"]].copy()
        # data["data-source"] = "PAMAP"
        labels = user_extract["activity"]
        print(f"{user} {data.shape}")
        pamap_all_data[user] = [(data, labels)]

    user_datasets = {'acc': pamap_acc_datasets,
                     'gyro': pamap_gyro_datasets, 
                     'mag': pamap_mag_datasets, 
                     'all': pamap_all_data}
    return user_datasets

def process_hhar_all_har_files(data_folder_path):
    """
    Preprocess all files of from the HHAR dataset
    Data files can be found at http://archive.ics.uci.edu/ml/datasets/heterogeneity+activity+recognition
    Args:
        data_folder_path: folder path

    Returns:
        user_datasets (a list containing data collected from the acc + gyro from both smartphones and watches)
    """
    har_phone_acc = pd.read_csv(os.path.join(data_folder_path, 'Phones_accelerometer.csv'))
    har_phone_acc['Device'] = "Phone acc"
    har_phone_gyro = pd.read_csv(os.path.join(data_folder_path, 'Phones_gyroscope.csv'))
    har_phone_gyro['Device'] = "Phone gyro"
    har_watch_acc = pd.read_csv(os.path.join(data_folder_path, 'Watch_accelerometer.csv'))
    har_watch_acc['Device'] = "Watch acc"
    har_watch_gyro = pd.read_csv(os.path.join(data_folder_path, 'Watch_gyroscope.csv'))
    har_watch_gyro['Device'] = "Watch gyro"
    acc_data = pd.concat([har_phone_acc, har_watch_acc])
    gyro_data = pd.concat([har_phone_gyro, har_watch_gyro])

    # acc_data.dropna(how="any", inplace=True)
    acc_data = acc_data[["x", "y", "z", "gt", "User", "Device"]]
    acc_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id", "device"]
    har_users = acc_data["user-id"].unique()

    acc_datasets = {}
    for user in har_users:
        user_extract = acc_data[acc_data["user-id"] == user]
        data = user_extract[["x-axis", "y-axis", "z-axis"]].copy()
        # data["data-source"] = "HHAR"
        labels = user_extract["activity"].values
        print(f"{user} {data.shape}")
        acc_datasets[user] = [(data, labels)]

    # gyro_data.dropna(how="any", inplace=True)
    gyro_data = gyro_data[["x", "y", "z", "gt", "User", "Device"]]
    gyro_data.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id", "device"]
    gyro_datasets = {}
    for user in har_users:
        user_extract = gyro_data[gyro_data["user-id"] == user]
        data = user_extract[["x-axis", "y-axis", "z-axis"]].copy()
        # data["data-source"] = "HHAR"
        labels = user_extract["activity"].values
        print(f"{user} {data.shape}")
        gyro_datasets[user] = [(data, labels)]

    all_data = {}
    gyro_acc_data = pd.concat([acc_data, gyro_data])
    gyro_acc_data.dropna(how="any", inplace=True)

    for user in har_users:
        user_extract = gyro_acc_data[gyro_acc_data["user-id"] == user]
        data = user_extract[["x-axis", "y-axis", "z-axis"]].copy()
        # data["data-source"] = "HHAR"
        labels = user_extract["activity"].values
        print(f"{user} {data.shape}")
        all_data[user] = [(data, labels)]

    user_datasets = {'acc': acc_datasets,
                     'gyro': gyro_datasets, 
                     'all': all_data}
    return user_datasets

def store_pickle(dataset, filename):
    with open(filename+'.pickle', 'wb') as file:
        pickle.dump(dataset, file)


def open_pickle(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)