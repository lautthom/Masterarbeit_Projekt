import pandas as pd
import pathlib
import numpy as np
import torch
import joblib


def save_samples(samples, sample_duration, classes):
    seconds, decimal = divmod(int(sample_duration*10), 10)
    with open(pathlib.Path(f'src/../data/processed/eda_samples_{seconds}_{decimal}s_classes_{min(classes)}_{max(classes)}.npy'), 'wb') as file:
        np.save(file, samples)


def save_samples_normalized(samples, sample_duration, classes):
    seconds, decimal = divmod(int(sample_duration*10), 10)
    with open(pathlib.Path(f'src/../data/processed/eda_samples_normalized_{seconds}_{decimal}s_classes_{min(classes)}_{max(classes)}.npy'), 'wb') as file:
        np.save(file, samples)


def save_labels(labels,classes):
    with open(pathlib.Path(f'src/../data/processed/labels_classes_{min(classes)}_{max(classes)}.npy'), 'wb') as file:
        np.save(file, labels)


def save_feature_vectors(feature_vectors, sample_duration, classes):
    seconds, decimal = divmod(int(sample_duration*10), 10)
    with open(pathlib.Path(f'src/../data/final/feature_vectors_{seconds}_{decimal}s_classes_{min(classes)}_{max(classes)}.npy'), 'wb') as file:
        np.save(file, feature_vectors)


def save_time_sequences_feature_vectors(time_sequences_feature_vectors, sample_duration, classes):
    seconds, decimal = divmod(int(sample_duration*10), 10)
    with open(pathlib.Path(f'src/../data/final/time_sequences_feature_vectors_{seconds}_{decimal}s_classes_{min(classes)}_{max(classes)}.npy'), 'wb') as file:
        np.save(file, time_sequences_feature_vectors)


def save_results(results_df, testing_mode):
    results_df.to_csv(pathlib.Path(f'src/../results/loso_results_{testing_mode}.csv'), index=False)


def get_subjects():
    subjects_df = pd.read_csv(pathlib.Path('src/../data/raw/PartC-Biosignals/samples.csv'), sep='\t')
    subjects = subjects_df.subject_name.tolist()
    return subjects


def load_samples(sample_duration, classes):
    seconds, decimal = divmod(int(sample_duration*10), 10)
    return np.load(pathlib.Path(f'src/../data/processed/eda_samples_{seconds}_{decimal}s_classes_{min(classes)}_{max(classes)}.npy'))


def load_samples_normalized(sample_duration, classes):
    seconds, decimal = divmod(int(sample_duration*10), 10)
    return np.load(pathlib.Path(f'src/../data/processed/eda_samples_normalized_{seconds}_{decimal}s_classes_{min(classes)}_{max(classes)}.npy'))

def load_labels(classes):
    return np.load(pathlib.Path(f'src/../data/processed/labels_classes_{min(classes)}_{max(classes)}.npy'))


def load_feature_vectors(sample_duration, classes):
    seconds, decimal = divmod(int(sample_duration*10), 10)
    return np.load(pathlib.Path(f'src/../data/final/feature_vectors_{seconds}_{decimal}s_classes_{min(classes)}_{max(classes)}.npy'))


def load_time_sequences_feature_vectors(sample_duration, classes):
    seconds, decimal = divmod(int(sample_duration*10), 10)
    return np.load(pathlib.Path(f'src/../data/final/time_sequences_feature_vectors_{seconds}_{decimal}s_classes_{min(classes)}_{max(classes)}.npy'))


def save_model(model, testing_mode, model_type, subject_index):
    torch.save(model, pathlib.Path(f'src/../models/{testing_mode}_{model_type}_{subject_index}.pth'))

def load_model(testing_mode, model_type, subject_index):
    return torch.load(pathlib.Path(f'src/../models/{testing_mode}_{model_type}_{subject_index}.pth'))

def save_forest(model, testing_mode, subject_index):
    joblib.dump(model, pathlib.Path(f'src/../models/{testing_mode}_forest_{subject_index}.joblib'))

def load_forest(testing_mode, subject_index):
    return joblib.load(pathlib.Path(f'src/../models/{testing_mode}_forest_{subject_index}.joblib'))