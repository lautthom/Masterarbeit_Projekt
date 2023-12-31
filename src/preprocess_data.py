import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import neurokit2 as nk
import pathlib


def get_cut_out_samples_and_labels(subjects, classes, sample_duration):

    datapoints_per_sample = int(round(sample_duration * 512, 0))
    
    samples = np.empty([0, 40, datapoints_per_sample])
    labels = np.empty([0, 40])

    for subject in subjects:
        data = pd.read_csv(pathlib.Path(f'src/../data/raw/PartC-Biosignals/biosignals_raw/{subject}.csv'), sep='\t')
        stimulus_data = pd.read_csv(pathlib.Path(f'src/../data/raw/PartC-Biosignals/stimulus/{subject}.csv'), sep='\t')

        data_filtered = data.filter(items=['time', 'gsr'])

        first_class_mask = stimulus_data['label'] == classes[0]
        second_class_mask = stimulus_data['label'] == classes[1]
        total_mask = first_class_mask | second_class_mask

        stimulus_data_labels = stimulus_data.loc[total_mask]
        labels_df = stimulus_data_labels.dropna()
        labels_array = labels_df.to_numpy()
        labels_data = np.squeeze(np.delete(labels_array, 0, 1))

        merged_data_time = pd.merge(data_filtered, stimulus_data_labels, on='time', how='left')
        merged_data = merged_data_time.drop(columns='time')

        merged_array = merged_data.to_numpy()

        stimulus_label_data = np.empty([0, datapoints_per_sample, merged_array[0].shape[0]])

        for index, item in enumerate(merged_array):
            if not np.isnan(item[1]):
                stimulus_label_data = np.append(stimulus_label_data, np.array([merged_array[index:index+datapoints_per_sample]]), axis=0)

        stimulus_data = np.squeeze(np.delete(stimulus_label_data, 1, 2))

        data = np.expand_dims(stimulus_data, axis=0)
        labels_proband = np.expand_dims(labels_data, axis=0)

        samples = np.append(samples, data, axis=0)
        labels = np.append(labels, labels_proband, axis=0)

    return samples, labels


def _process_eda_signal(sample):
    try: 
        signals, info = nk.eda_process(sample, sampling_rate=512)
        return signals['EDA_Clean'], signals['EDA_Tonic'].to_numpy(), signals['EDA_Phasic'].to_numpy()
    except ValueError:  # neurokit throws ValueError, when EDA signal is flat; since a flat signal does not need to be cleaned and has no phasic component, the existing signal will be used as cleaned and tonic signal, a flat line with value 0 will be used as phasic signal.
        return sample, sample, np.zeros([sample.shape[0]])
    

def compute_tonic_and_phasic_components(data):
    processed_proband_data = np.empty([0, data.shape[1], data.shape[2], 3])
    for proband in data:
        processed_samples = np.empty([0, data.shape[2], 3])
        for sample in proband:
            processed_eda_signal = np.stack((_process_eda_signal(sample)), axis=1)
            processed_samples = np.append(processed_samples, np.expand_dims(processed_eda_signal, axis=0), axis=0)
        processed_proband_data = np.append(processed_proband_data, np.expand_dims(processed_samples, axis=0), axis=0)
    return processed_proband_data

    

def _compute_feature_vector(sample):
    clean_features = _compute_statistical_descriptors(pd.Series(sample[:, 0]))
    tonic_features = _compute_statistical_descriptors(pd.Series(sample[:, 1]))
    phasic_features = _compute_statistical_descriptors(pd.Series(sample[:, 2]))
    feature_vector = np.concatenate((clean_features, tonic_features, phasic_features), axis=1)
    return feature_vector


def _compute_statistical_descriptors(signal):
    max = signal.max()
    min = signal.min()
    mean = signal.mean()
    std = signal.std()
    var = signal.var() 
    rms = math.sqrt(sum(signal**2) / len(signal))
    p2p = max - min
    skew = signal.skew()
    kurtosis = signal.kurtosis()

    first_diff = signal.diff()
    mean_first_diff = first_diff.mean()
    mean_abs_first_diff = first_diff.abs().mean()
    mean_abs_second_diff = first_diff.diff().abs().mean() 
    return np.array([[max, min, mean, std, var, rms, p2p, skew, kurtosis, mean_first_diff, mean_abs_first_diff, mean_abs_second_diff]])


def compute_feature_vectors(data):
    feature_vectors = np.empty([0, data.shape[1], 36])
    for proband in data:
        proband_vector = np.empty([0, 36])
        for sample in proband:
            proband_vector = np.append(proband_vector, _compute_feature_vector(sample), axis=0)
        proband_vector = np.expand_dims(proband_vector, axis=0)
        feature_vectors = np.append(feature_vectors, proband_vector, axis=0)
    return feature_vectors


def compute_time_sequences_feature_vectors(data):
    number_feature_vectors = int(data.shape[2] // 512)
    complete_time_sequence_features = np.empty([0, data.shape[1], number_feature_vectors, 36])
    for proband in data:
        probands_features = np.empty([0, number_feature_vectors, 36])
        for sample in proband:
            features_samples = np.empty([0, 36])
            for i in range(number_feature_vectors):
                features_time_sequence = _compute_feature_vector(sample[i*512 : (i+1)*512])
                features_samples = np.append(features_samples, features_time_sequence, axis=0)
            probands_features = np.append(probands_features, np.expand_dims(features_samples, axis=0), axis=0)
        complete_time_sequence_features = np.append(complete_time_sequence_features, np.expand_dims(probands_features, axis=0), axis=0)
    return complete_time_sequence_features


def reduce_eda_signal(data):  
    complete_means = np.empty([0, data.shape[1], data.shape[2]//32, 3])
    for proband in data:
        proband_means = np.empty([0, data.shape[2]//32, 3])
        for sample in proband:
            means = np.empty([0, 3])
            for i in range(data.shape[2] // 32):
                mean = np.mean(sample[i*32:(i+1)*32], axis=0)
                means = np.append(means, np.expand_dims(mean, axis=0), axis=0)
            proband_means = np.append(proband_means, np.expand_dims(means, axis=0), axis=0)
        complete_means = np.append(complete_means, np.expand_dims(proband_means, axis=0), axis=0)
    return complete_means