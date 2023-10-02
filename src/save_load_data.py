import pandas as pd
import pathlib
import numpy as np


def save_samples(samples, sample_duration):
    seconds, decimal = divmod(int(sample_duration*10), 10)
    with open(pathlib.Path(f'src/../data/processed/eda_samples_{seconds}_{decimal}.npy'), 'wb') as file:
        np.save(file, samples)


def save_samples_normalized(samples, sample_duration):
    seconds, decimal = divmod(int(sample_duration*10), 10)
    with open(pathlib.Path(f'src/../data/processed/eda_samples_normalized_{seconds}_{decimal}.npy'), 'wb') as file:
        np.save(file, samples)


def save_labels(labels):
    with open(pathlib.Path(f'src/../data/processed/labels.npy'), 'wb') as file:
        np.save(file, labels)


def save_feature_vectors(feature_vectors, sample_duration):
    seconds, decimal = divmod(int(sample_duration*10), 10)
    with open(pathlib.Path(f'src/../data/final/feature_vectors_{seconds}_{decimal}.npy'), 'wb') as file:
        np.save(file, feature_vectors)


def save_time_sequences_feature_vectors(time_sequences_feature_vectors, sample_duration):
    seconds, decimal = divmod(int(sample_duration*10), 10)
    with open(pathlib.Path(f'src/../data/final/time_sequences_feature_vectors_{seconds}_{decimal}.npy'), 'wb') as file:
        np.save(file, time_sequences_feature_vectors)


def get_subjects():
    subjects_df = pd.read_csv(pathlib.Path('src/../data/raw/PartC-Biosignals/samples.csv'), sep='\t')
    subjects = subjects_df.subject_name.tolist()
    return subjects


def load_samples(sample_duration):
    seconds, decimal = divmod(int(sample_duration*10), 10)
    return np.load(pathlib.Path(f'src/../data/processed/eda_samples_{seconds}_{decimal}.npy'))


def load_samples_normalized(sample_duration):
    seconds, decimal = divmod(int(sample_duration*10), 10)
    return np.load(pathlib.Path(f'src/../data/processed/eda_samples_normalized_{seconds}_{decimal}.npy'))

def load_labels():
    return np.load(pathlib.Path(f'src/../data/processed/labels.npy'))


def load_feature_vectors(sample_duration):
    seconds, decimal = divmod(int(sample_duration*10), 10)
    return np.load(pathlib.Path(f'src/../data/final/feature_vectors_{seconds}_{decimal}.npy'))


def load_time_sequences_feature_vectors(sample_duration):
    seconds, decimal = divmod(int(sample_duration*10), 10)
    return np.load(pathlib.Path(f'src/../data/final/time_sequences_feature_vectors_{seconds}_{decimal}.npy'))