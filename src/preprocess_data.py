import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import neurokit2 as nk
import pathlib


def _save_data_proband(data_eda, data_labels, subject):
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


def get_cut_out_data(subjects):
    for subject in subjects:
        data = pd.read_csv(pathlib.Path(f'src/../data/raw/PartC-Biosignals/biosignals_raw/{subject}.csv'), sep='\t')
        stimulus_data = pd.read_csv(pathlib.Path(f'src/../data/raw/PartC-Biosignals/stimulus/{subject}.csv'), sep='\t')

        data_filtered = data.filter(items=['time', 'gsr'])

        # possible refactoring
        mask_label_1 = stimulus_data['label'] == 1 
        mask_label_4 = stimulus_data['label'] == 4
        total_mask = mask_label_1 | mask_label_4

        stimulus_data_labels = stimulus_data.loc[total_mask]
        labels_df = stimulus_data_labels.dropna()
        labels_array = labels_df.to_numpy()
        labels_data = np.squeeze(np.delete(labels_array, 0, 1))

        merged_data_time = pd.merge(data_filtered, stimulus_data_labels, on='time', how='left')
        merged_data = merged_data_time.drop(columns='time')

        merged_array = merged_data.to_numpy()

        time_in_s = 8
        number_of_samples = time_in_s * 512

        datapoints_per_sample = merged_array[0].shape[0]

        stimulus_label_data = np.empty([0, number_of_samples, datapoints_per_sample])

        for index, item in enumerate(merged_array):
            if not np.isnan(item[1]):
                stimulus_label_data = np.append(stimulus_label_data, np.array([merged_array[index:index+number_of_samples]]), axis=0)

        stimulus_data = np.squeeze(np.delete(stimulus_label_data, 1, 2))

        data = np.expand_dims(stimulus_data, axis=0)
        labels = np.expand_dims(labels_data, axis=0)

        _save_data_proband(data, labels, subject)


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


def _compute_feature_vector(sample):
    try:
        signals, info = nk.eda_process(sample, sampling_rate=50)
        clean_features = _compute_statistical_descriptors(signals['EDA_Clean'])
        tonic_features = _compute_statistical_descriptors(signals['EDA_Tonic'])
        phasic_features = _compute_statistical_descriptors(signals['EDA_Phasic'])
    except ValueError:  # neurokit throws ValueError, when EDA signal is flat; since a flat signal does not need to be cleaned and has no phasic component, the existing signal will be used as cleaned and tonic signal, a flat line with value 0 will be used as phasic signal.      
        sample = pd.Series(sample)
        clean_features = _compute_statistical_descriptors(sample)
        tonic_features = _compute_statistical_descriptors(sample)
        flat_phasic_signal = pd.Series(0, index = np.arange(8 * 512))
        phasic_features = _compute_statistical_descriptors(flat_phasic_signal)
    # mean_peak_amplitude = np.array([[info['SCR_Amplitude'].mean()]])
    # feature_vector = np.concatenate((clean_features, tonic_features, phasic_features, mean_peak_amplitude), axis=1)
    feature_vector = np.concatenate((clean_features, tonic_features, phasic_features), axis=1)
    return feature_vector


def _compute_statistical_descriptors(signal):
    max = signal.max()
    min = signal.min()
    mean = signal.mean()
    std = signal.std()
    var = signal.var()  # necessary?
    rms = math.sqrt(sum(signal**2) / len(signal))
    # power ?  spectral power?
    # peak ?  SCR_Peaks/SCR_Amplitude according to Neurokit info? applicable for several peaks? highest peaks? peak count? mean peak amplitude? sum peak amplitude?
    p2p = max - min
    skew = signal.skew()
    kurtosis = signal.kurtosis()
    # crest_factor ?  mutliple peaks?
    # form_factor ?
    # pulse_indicator ? not relevant for eda/wrong for eda
    # var_second_moment ? second moment is scalar, how is variation(?) calculated? Also, rms square root of second moment?
    # variation_second_moment ? what is variation? second moment scalar?
    # std_second_moment ? scalar? redundant due to var_second_moment?

    first_diff = signal.diff()
    mean_first_diff = first_diff.mean()
    mean_abs_first_diff = first_diff.abs().mean()
    mean_abs_second_diff = first_diff.diff().abs().mean()  # without abs() ?
    return np.array([[max, min, mean, std, var, rms, p2p, skew, kurtosis, mean_first_diff, mean_abs_first_diff, mean_abs_second_diff]])


def compute_feature_vectors(data):
    number_of_samples = 40

    feature_vectors = np.empty([0, number_of_samples, 36])
    for proband in data:
        proband_vector = np.empty([0, 36])
        for sample in proband:
            proband_vector = np.append(proband_vector, _compute_feature_vector(sample), axis=0)
        proband_vector = np.expand_dims(proband_vector, axis=0)
        feature_vectors = np.append(feature_vectors, proband_vector, axis=0)
    return feature_vectors


def cut_out_time_sequences(data):  
    complete_time_sequence_features = np.empty([0, 40, 8, 12])
    for proband in data:
        probands_features = np.empty([0, 8, 12])
        for sample in proband:
            features_samples = np.empty([0, 12])
            for i in range(8):
                features_time_sequence = _compute_statistical_descriptors(pd.Series(sample[i * 512:(i+1) * 512]))
                features_samples = np.append(features_samples, features_time_sequence, axis=0)
            probands_features = np.append(probands_features, np.expand_dims(features_samples, axis=0), axis=0)
        complete_time_sequence_features = np.append(complete_time_sequence_features, np.expand_dims(probands_features, axis=0), axis=0)
    return complete_time_sequence_features