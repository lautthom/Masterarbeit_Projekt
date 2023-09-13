import pandas as pd
import pathlib
import numpy as np


def save_data_proband(data_eda, data_labels, subject):
    with open(pathlib.Path(f'src/../data/processed/{subject}_eda.npy'), 'wb') as file:
        np.save(file, data_eda)
    with open(pathlib.Path(f'src/../data/processed/{subject}_labels.npy'), 'wb') as file:
        np.save(file, data_labels)


def save_feature_vectors(feature_vectors):
    with open(pathlib.Path(f'src/../data/final/feature_vectors.npy'), 'wb') as file:
        np.save(file, feature_vectors)


def get_subjects():
    subjects_df = pd.read_csv(pathlib.Path('src/../data/raw/PartC-Biosignals/samples.csv'), sep='\t')
    subjects = subjects_df.subject_name.tolist()
    return subjects


def load_data(subjects):  
    time_in_s = 8
    number_of_samples = 40
    timepoints = time_in_s * 512

    data_train = np.empty([0, number_of_samples, timepoints])
    labels_train = np.empty([0, number_of_samples])

    for subject in subjects:
        subject_data = np.load(pathlib.Path(f'src/../data/processed/{subject}_eda.npy'))
        subject_labels = np.load(pathlib.Path(f'src/../data/processed/{subject}_labels.npy'))
            
        data_train = np.append(data_train, subject_data, axis=0)
        labels_train = np.append(labels_train, subject_labels, axis=0)

    return (data_train, labels_train)


def load_feature_vectors():
    return np.load(pathlib.Path(f'src/../data/final/feature_vectors.npy'))