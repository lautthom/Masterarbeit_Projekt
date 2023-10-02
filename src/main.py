import preprocess_data
import save_load_data
import numpy as np
import baseline_models
import sklearn
import deep_model


def get_data(subjects, do_preprocessing, classes, sample_duration):
    if do_preprocessing:
        # TODO: add options for possible other options
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
    return data_eda, feature_vectors_eda, time_sequences_feature_vectors, labels


def loso(data_eda, feature_vectors_eda, time_sequences_feature_vectors, labels, subjects, classes, learning_rates, num_epochs, batch_size, hidden_state_sizes, num_recurrent_layers, use_grus):
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
        sequence_feature_vectors_training = sequence_feature_vectors_training.reshape((86 * 40, sequence_feature_vectors_training.shape[2], 36))
        labels_training = labels_training.ravel()

        data_training, feature_vectors_training, sequence_feature_vectors_training, labels_training = sklearn.utils.shuffle(data_training, feature_vectors_training, sequence_feature_vectors_training, labels_training)
        data_test, feature_vectors_test, sequence_feature_vectors_test, labels_test = sklearn.utils.shuffle(data_test, feature_vectors_test, sequence_feature_vectors_test, labels_test)

        print('Running Random Forest...')
        accuracy_forest = baseline_models.random_forest_model(feature_vectors_training, labels_training, feature_vectors_test, labels_test)
        print(f'Random Forest Accuracy: {accuracy_forest}')
        random_forest_accuracies.append(accuracy_forest)
        print(' ')

        #Make enum (or comparable) for type of model

        print('Running RNN model...')
        accuracy_rnn = deep_model.run_model('rnn', data_training, labels_training, data_test, labels_test, classes, learning_rates['rnn'], num_epochs['rnn'], batch_size, hidden_state_sizes['rnn'], num_recurrent_layers['rnn'], use_grus)
        print(f'RNN Model Accuracy: {accuracy_rnn}')
        rnn_accuracies.append(accuracy_rnn)
        print(' ')

        print('Running CRNN model...')
        accuracy_crnn = deep_model.run_model('crnn', data_training, labels_training, data_test, labels_test, classes, learning_rates['crnn'], num_epochs['crnn'], batch_size, hidden_state_sizes['crnn'], num_recurrent_layers['crnn'], use_grus)
        print(f'CRNN Model Accuracy: {accuracy_crnn}')
        crnn_accuracies.append(accuracy_crnn)
        print(' ')

        print('Running Feature RNN model...')
        accuracy_feature_rnn = deep_model.run_model('feature_rnn', sequence_feature_vectors_training, labels_training, sequence_feature_vectors_test, labels_test, classes, learning_rates['feature_rnn'], num_epochs['feature_rnn'], batch_size, hidden_state_sizes['feature_rnn'], num_recurrent_layers['rnn'], use_grus)
        print(f'Feature RNN Model Accuracy: {accuracy_feature_rnn}')
        feature_rnn_accuracies.append(accuracy_feature_rnn)
        print(' ')

        print('Running CNN model...')
        accuracy_cnn = deep_model.run_model('cnn', data_training, labels_training, data_test, labels_test, classes, learning_rates['cnn'], num_epochs['cnn'], batch_size)
        print(f'CNN Model Accuracy: {accuracy_cnn}')
        cnn_accuracies.append(accuracy_cnn)
        print(' ')
    
    # TODO: implement confusion matrix for results
    print(f'Mean Random Forest model Accuracy: {sum(random_forest_accuracies) / len(random_forest_accuracies):.4f}')
    print(f'Mean RNN model Accuracy: {sum(rnn_accuracies) / len(rnn_accuracies):.4f}')
    print(f'Mean CRNN model Accuracy: {sum(crnn_accuracies) / len(crnn_accuracies):.4f}')
    print(f'Mean feature RNN model Accuracy: {sum(feature_rnn_accuracies) / len(feature_rnn_accuracies):.4f}')
    print(f'Mean CNN model Accuracy: {sum(cnn_accuracies) / len(cnn_accuracies):.4f}')


def grid_search():
    pass


def main():
    classes = (1, 4)
    sample_duration = round(8.55, 1)
    do_preprocessing = False
    batch_size = 40
    do_loso_run = True
    do_grid_search = False
    learning_rates = {'rnn': 0.1, 'crnn': 0.1, 'feature_rnn': 0.1, 'cnn': 0.005}
    num_epochs = {'rnn': 100, 'crnn': 100, 'feature_rnn': 100, 'cnn': 25}
    hidden_state_sizes = {'rnn': 128, 'crnn': 128, 'feature_rnn': 128}
    num_recurrent_layers = {'rnn': 2, 'crnn': 2, 'feature_rnn': 2}
    use_grus = False
    # implement verbose output of deep models or not
    # implement option for individual confusion matrices
    # implement option for overall confusion matrices
    # decice what to do with training plots
    # add list or dict for all deep model options?

    subjects = save_load_data.get_subjects()
    
    data_eda, feature_vectors_eda, time_sequences_feature_vectors, labels = get_data(subjects, do_preprocessing, classes, sample_duration)
    
    if do_loso_run:
        loso(data_eda, feature_vectors_eda, time_sequences_feature_vectors, labels, subjects, classes, learning_rates, num_epochs, batch_size, hidden_state_sizes, num_recurrent_layers, use_grus)

    if do_grid_search:
        grid_search(data_eda, feature_vectors_eda, time_sequences_feature_vectors, labels)

if __name__ == '__main__':
    # TODO: make .txt file of dependencies and order dependencies correctly
    # TODO: write documentation
    # TODO: implement grid search for hyperparameters
    # TODO: implement CLI
    main()