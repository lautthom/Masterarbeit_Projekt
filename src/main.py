import preprocess_data
import save_load_data
import numpy as np
import baseline_models
import sklearn
import deep_model


def get_data(subjects, preprocess, classes, sample_duration):
    if preprocess:
        # TODO: add options for classes, and possible other options
        # TODO: add all (correct) feature for feature vectors
        # TODO: check if feature vectors can be calculated faster (multithreading, applying functions to arrays without for-loop, etc.)
        print('Getting data and cutting out samples...')
        data_eda, labels = preprocess_data.get_cut_out_samples_and_labels(subjects, classes, sample_duration)
        #### data_eda_normalized doesn't work!!!
        data_eda_normalized = data_eda.copy()
        data_eda_normalized = data_eda_normalized / np.max(data_eda_normalized)
        data_eda_signal_tonic_phasic = preprocess_data.compute_tonic_and_phasic_components(data_eda)
        data_eda = data_eda_signal_tonic_phasic

        print('Computing feature vectors...')
        feature_vectors_eda = preprocess_data.compute_feature_vectors(data_eda)
        # TODO: option for half second feature vector? if yes, also change file saving/loading system for time_sequence_feature_vectors
        time_sequences_feature_vectors = preprocess_data.compute_time_sequences_feature_vectors(data_eda)
       
        data_eda = preprocess_data.reduce_eda_signal(data_eda)
        
        save_load_data.save_samples(data_eda, sample_duration, classes)
        save_load_data.save_samples_normalized(data_eda_normalized, sample_duration, classes)
        save_load_data.save_labels(labels, classes)   
        save_load_data.save_feature_vectors(feature_vectors_eda, sample_duration, classes)
        save_load_data.save_time_sequences_feature_vectors(time_sequences_feature_vectors, sample_duration, classes)
        
    else:
        print('Loading data and feature vectors...')
        data_eda = save_load_data.load_samples(sample_duration, classes)
        data_eda_normalized = save_load_data.load_samples_normalized(sample_duration, classes)
        labels = save_load_data.load_labels(classes)
        feature_vectors_eda = save_load_data.load_feature_vectors(sample_duration, classes)
        time_sequences_feature_vectors = save_load_data.load_time_sequences_feature_vectors(sample_duration, classes)

    #### data_eda_normalized doesn't work!!!
    return data_eda, labels, feature_vectors_eda, time_sequences_feature_vectors


def main():
    classes = (2, 3)
    sample_duration = round(8.55, 1)
    preprocess = False
    batch_size = 40
    
    subjects = save_load_data.get_subjects()
    
    data_eda, labels, feature_vectors_eda, time_sequences_feature_vectors = get_data(subjects, preprocess, classes, sample_duration)

    random_forest_accuracies = []
    rnn_accuracies = []
    crnn_accuracies = []
    feature_rnn_accuracies = []
    cnn_accuracies = []

    # TODO: add option to save/load models
    # TODO: add options for batch size, learning rate, etc.
    
    # TODO: add GRU unit for RNN models?

    for index, (proband_data, proband_feature_vectors, proband_sequence_feature_vectors, proband_labels, proband_name) in enumerate(zip(data_eda, feature_vectors_eda, time_sequences_feature_vectors, labels, subjects)):
        print(f'Current proband: {proband_name}')
        data_test = proband_data
        feature_vectors_test = proband_feature_vectors
        sequence_feature_vectors_test = proband_sequence_feature_vectors
        labels_test = proband_labels
        
        data_training = np.delete(data_eda, np.s_[index], axis=0)
        feature_vectors_training = np.delete(feature_vectors_eda, np.s_[index], 0)
        sequence_feature_vectors_training = np.delete(time_sequences_feature_vectors, np.s_[index], 0)
        labels_training = np.delete(labels, np.s_[index], 0)
        data_training = data_training.reshape((86 * 40, data_training.shape[2], 3))
        feature_vectors_training = feature_vectors_training.reshape((86 * 40, 36))
        sequence_feature_vectors_training = sequence_feature_vectors_training.reshape((86 * 40, int(sample_duration), 36))
        labels_training = labels_training.ravel()

        data_training, feature_vectors_training, sequence_feature_vectors_training, labels_training = sklearn.utils.shuffle(data_training, feature_vectors_training, sequence_feature_vectors_training, labels_training)
        data_test, feature_vectors_test, sequence_feature_vectors_test, labels_test = sklearn.utils.shuffle(data_test, feature_vectors_test, sequence_feature_vectors_test, labels_test)

        print('Running Random Forest...')
        accuracy_forest = baseline_models.random_forest_model(feature_vectors_training, labels_training, feature_vectors_test, labels_test)
        print(f'Random Forest Accuracy: {accuracy_forest}')
        random_forest_accuracies.append(accuracy_forest)

        #Make enum (or comparable) for type of model

        # print('Running RNN model...')
        # accuracy_rnn = deep_model.run_model('rnn', data_training, labels_training, data_test, labels_test, batch_size)
        # print(f'RNN Model Accuracy: {accuracy_rnn}')
        # rnn_accuracies.append(accuracy_rnn)

        print('Running CRNN model...')
        accuracy_crnn = deep_model.run_model('crnn', data_training, labels_training, data_test, labels_test, batch_size, classes)
        print(f'CRNN Model Accuracy: {accuracy_crnn}')
        crnn_accuracies.append(accuracy_crnn)

        # print('Running Feature RNN model...')
        # accuracy_feature_rnn = deep_model.run_model('feature_rnn', sequence_feature_vectors_training, labels_training, sequence_feature_vectors_test, labels_test, batch_size)
        # print(f'Feature RNN Model Accuracy: {accuracy_feature_rnn}')
        # feature_rnn_accuracies.append(accuracy_feature_rnn)

        print('Running CNN model...')
        accuracy_cnn = deep_model.run_model('cnn', data_training, labels_training, data_test, labels_test, batch_size, classes)
        print(f'CNN Model Accuracy: {accuracy_cnn}')
        cnn_accuracies.append(accuracy_cnn)
        print(' ')
    
    # TODO: implement confusion matrix for results
    print(f'Mean Random Forest model Accuracy: {sum(random_forest_accuracies) / len(random_forest_accuracies):.4f}')
    print(f'Mean RNN model Accuracy: {sum(rnn_accuracies) / len(rnn_accuracies):.4f}')
    print(f'Mean CRNN model Accuracy: {sum(crnn_accuracies) / len(crnn_accuracies):.4f}')
    print(f'Mean feature RNN model Accuracy: {sum(feature_rnn_accuracies) / len(feature_rnn_accuracies):.4f}')
    print(f'Mean CNN model Accuracy: {sum(cnn_accuracies) / len(cnn_accuracies):.4f}')
        


if __name__ == '__main__':
    # TODO: make .txt file of dependencies and order dependencies correctly
    # TODO: write documentation
    # TODO: implement grid search for hyperparameters
    # TODO: implement CLI
    main()