'''
This file is used to pre-process all of the raw data and turn them into pickled files to be used. 
This should only be run once
'''

from raw_data_processing import process_hhar_all_files, process_hhar_all_har_files, process_motion_sense_all_files, process_motion_sense_all_har_files, process_PAMAP2_all_data, process_pamap2_har_files, process_HARTH_thigh_acc_har_data, process_HARTH_all_acc_data
import pickle

if __name__ == '__main__':
    pamap_df = process_PAMAP2_all_data("test_run/original_datasets/PAMAP2")
    hhar_df = process_hhar_all_files("test_run/original_datasets/hhar/Activity recognition exp")
    motion_sense_df = process_motion_sense_all_files("test_run/original_datasets/motionsense/Data/")
    harth_df = process_HARTH_all_acc_data("test_run/original_datasets/harth")

    # pickle the datasets
    with open('pickled_datasets/pamap2.pickle', 'wb') as file:
        pickle.dump(pamap_df, file)
    with open('pickled_datasets/hhar2.pickle', 'wb') as file:
        pickle.dump(hhar_df, file)
    with open('pickled_datasets/motionsense2.pickle', 'wb') as file:
        pickle.dump(motion_sense_df, file)
    with open('pickled_datasets/harth2.pickle', 'wb') as file:
        pickle.dump(harth_df, file)
    
    pamap_har_df = process_pamap2_har_files("test_run/original_datasets/PAMAP2")
    hhar_har_df = process_hhar_all_har_files("test_run/original_datasets/hhar/Activity recognition exp")
    motion_sense_har_df = process_motion_sense_all_har_files("test_run/original_datasets/motionsense/Data/")
    harth_har_df = process_HARTH_thigh_acc_har_data("test-run/original_datasets/harth")

    with open('pickled_datasets/pamap_har.pickle', 'wb') as file:
        pickle.dump(pamap_har_df, file)
    with open('pickled_datasets/hhar_har.pickle', 'wb') as file:
        pickle.dump(hhar_har_df, file)
    with open('pickled_datasets/motionsense_har.pickle', 'wb') as file:
        pickle.dump(motion_sense_har_df, file)
    with open('pickled_datasets/harth_har.pickle', 'wb') as file:
        pickle.dump(harth_har_df, file)