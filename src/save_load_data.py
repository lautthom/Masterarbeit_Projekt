import pandas as pd
import pathlib
import numpy as np


def save_samples(samples):
    with open(pathlib.Path(f'src/../data/processed/eda_samples.npy'), 'wb') as file:
        np.save(file, samples)


def save_labels(labels):
    with open(pathlib.Path(f'src/../data/processed/labels.npy'), 'wb') as file:
        np.save(file, labels)


def save_feature_vectors(feature_vectors):
    with open(pathlib.Path(f'src/../data/final/feature_vectors.npy'), 'wb') as file:
        np.save(file, feature_vectors)


def save_time_sequences_feature_vectors(time_sequences_feature_vectors):
    with open(pathlib.Path(f'src/../data/final/time_sequences_feature_vectors.npy'), 'wb') as file:
        np.save(file, time_sequences_feature_vectors)


def get_subjects():
    subjects_df = pd.read_csv(pathlib.Path('src/../data/raw/PartC-Biosignals/samples.csv'), sep='\t')
    subjects = subjects_df.subject_name.tolist()
    return subjects


def load_samples():
    return np.load(pathlib.Path(f'src/../data/processed/eda_samples.npy'))


def load_labels():
    return np.load(pathlib.Path(f'src/../data/processed/labels.npy'))


def load_feature_vectors():
    return np.load(pathlib.Path(f'src/../data/final/feature_vectors.npy'))


def load_time_sequences_feature_vectors():
    return np.load(pathlib.Path(f'src/../data/final/time_sequences_feature_vectors.npy'))