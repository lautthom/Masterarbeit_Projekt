import preprocess_data
import save_load_data
import numpy as np
import baseline_models
import sklearn
import deep_model
import torch
import pandas as pd


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
        data_eda_signal_tonic_phasic_normalized = preprocess_data.compute_tonic_and_phasic_components(data_eda_normalized)

        print('Computing feature vectors...')
        feature_vectors_eda = preprocess_data.compute_feature_vectors(data_eda_signal_tonic_phasic)
        # TODO: option for half second feature vector? if yes, also change file saving/loading system for time_sequence_feature_vectors
        time_sequences_feature_vectors = preprocess_data.compute_time_sequences_feature_vectors(data_eda_signal_tonic_phasic)
       
        data_eda = preprocess_data.reduce_eda_signal(data_eda_signal_tonic_phasic)
        
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
    
    # TODO: add GRU unit for RNN models?
    for index, (proband_data, proband_feature_vectors, proband_sequence_feature_vectors, proband_labels, proband_name) in enumerate(zip(data_eda, feature_vectors_eda, time_sequences_feature_vectors, labels, subjects)):
        print(' ')
        print(f'Current proband: {proband_name}')
        data_test = proband_data
        feature_vectors_test = proband_feature_vectors
        sequence_feature_vectors_test = proband_sequence_feature_vectors
        labels_test = proband_labels
        
        data_training = np.delete(data_eda, np.s_[index], axis=0)
        feature_vectors_training = np.delete(feature_vectors_eda, np.s_[index], 0)
        sequence_feature_vectors_training = np.delete(time_sequences_feature_vectors, np.s_[index], 0)
        labels_training = np.delete(labels, np.s_[index], 0)
        # might want to replace 40 with .shape[1], and 3 or 36 with .shape[3]
        data_training = data_training.reshape((data_training.shape[0] * 40, data_training.shape[2], 3))
        feature_vectors_training = feature_vectors_training.reshape((feature_vectors_training.shape[0] * 40, 36))
        sequence_feature_vectors_training = sequence_feature_vectors_training.reshape((sequence_feature_vectors_training.shape[0] * 40, sequence_feature_vectors_training.shape[2], 36))
        labels_training = labels_training.ravel()

        # weglassen? entscheiden, wo geshuffled wird  
        data_training, feature_vectors_training, sequence_feature_vectors_training, labels_training = sklearn.utils.shuffle(data_training, feature_vectors_training, sequence_feature_vectors_training, labels_training)
        data_test, feature_vectors_test, sequence_feature_vectors_test, labels_test = sklearn.utils.shuffle(data_test, feature_vectors_test, sequence_feature_vectors_test, labels_test)

        print('Running Random Forest...')
        accuracy_forest = baseline_models.random_forest_model(feature_vectors_training, labels_training, feature_vectors_test, labels_test)
        print(f'Random Forest Accuracy: {accuracy_forest}')
        random_forest_accuracies.append(accuracy_forest)
        print(' ')

        #Make enum (or comparable) for type of model

        print('Running RNN model...')
        accuracy_rnn = deep_model.run_model('rnn', data_training, labels_training, data_test, labels_test, classes, learning_rates['rnn'], num_epochs['rnn'], batch_size, hidden_state_sizes['rnn'], num_recurrent_layers['rnn'], use_grus=use_grus)
        print(f'RNN Model Accuracy: {accuracy_rnn}')
        rnn_accuracies.append(accuracy_rnn)
        print(' ')

        print('Running CRNN model...')
        accuracy_crnn = deep_model.run_model('crnn', data_training, labels_training, data_test, labels_test, classes, learning_rates['crnn'], num_epochs['crnn'], batch_size, hidden_state_sizes['crnn'], num_recurrent_layers['crnn'], use_grus=use_grus)
        print(f'CRNN Model Accuracy: {accuracy_crnn}')
        crnn_accuracies.append(accuracy_crnn)
        print(' ')

        print('Running Feature RNN model...')
        accuracy_feature_rnn = deep_model.run_model('feature_rnn', sequence_feature_vectors_training, labels_training, sequence_feature_vectors_test, labels_test, classes, learning_rates['feature_rnn'], num_epochs['feature_rnn'], batch_size, hidden_state_sizes['feature_rnn'], num_recurrent_layers['rnn'], use_grus=use_grus)
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

    results_df = pd.DataFrame(list(zip(subjects, random_forest_accuracies, rnn_accuracies, crnn_accuracies, feature_rnn_accuracies, cnn_accuracies)), columns=['Subjects', 'Accuracy Random Forest', 'Accuracy RNN', 'Accuracy CRNN', 'Accuracy Feature RNN', 'Accuracy CNN'])
    save_load_data.save_results(results_df)


def grid_search(data_eda, time_sequences_feature_vectors, labels, classes):
    data, time_sequences_feature_vectors, labels = sklearn.utils.shuffle(data_eda, time_sequences_feature_vectors, labels)
    
    data_train, data_test = np.split(data, [int(data.shape[0]*0.8)])
    time_sequences_feature_vectors_train, time_sequences_feature_vectors_test = np.split(time_sequences_feature_vectors, [int(data.shape[0]*0.8)])
    labels_train, labels_test = np.split(labels, [int(data.shape[0]*0.8)])

    data_train = data_train.reshape((data_train.shape[0] * 40, data_train.shape[2], 3))
    data_test = data_test.reshape((data_test.shape[0] * 40, data_test.shape[2], 3))
    time_sequences_feature_vectors_train = time_sequences_feature_vectors_train.reshape((time_sequences_feature_vectors_train.shape[0] * 40, time_sequences_feature_vectors_train.shape[2], 36))
    time_sequences_feature_vectors_test = time_sequences_feature_vectors_test.reshape((time_sequences_feature_vectors_test.shape[0] * 40, time_sequences_feature_vectors_test.shape[2], 36))
    
    labels_train = labels_train.ravel()
    labels_test = labels_test.ravel()

                                         
    # learning_rates = [0.0001]
    # epochs = [100]
    # num_recurrent_layers = [2] 
    # hidden_states = [64]

    # for epoch in epochs:
    #     for learning_rate in learning_rates:
    #         for num_recurrent_layer in num_recurrent_layers:
    #             for hidden_state in hidden_states:
    #                 accuracy = deep_model.run_model('rnn', data_train, labels_train, data_test, labels_test, classes, learning_rate, epoch, 40, hidden_state, num_recurrent_layer, use_grus=False)
    #                 print(f'For RNN model using {num_recurrent_layer} layer(s) with {hidden_state} hidden states, {learning_rate} learning rate, {epoch} epochs: accuracy {accuracy:.4f}')

    # learning_rates = [0.01, 0.05, 0.1]
    # epochs = [75, 100, 125, 150]
    # num_recurrent_layers = [1, 2, 3] 

    # for epoch in epochs:
    #     for learning_rate in learning_rates:
    #         for num_recurrent_layer in num_recurrent_layers:
    #             accuracy = deep_model.run_model('feature_rnn', time_sequences_feature_vectors_train, labels_train, time_sequences_feature_vectors_test, labels_test, classes, learning_rate, epoch, 40, 128, num_recurrent_layer, use_grus=False)
    #             print(f'For Feature RNN model using {num_recurrent_layer} layer(s), {learning_rate} learning rate, {epoch} epochs: accuracy {accuracy:.4f}')
 
    epochs = [100]
    learning_rates = [0.0001]
    kernel_sizes = [5, 7, 9, 11, 13, 15]

    for epoch in epochs:
        for learning_rate in learning_rates:
            for kernel_size in kernel_sizes:
                accuracy = deep_model.run_model('cnn', data_train, labels_train, data_test, labels_test, classes, learning_rate, epoch, 40, kernel_size=kernel_size)
                print(f'For CNN model using kernel size {kernel_size}, {learning_rate} learning rate, {epoch} epochs: accuracy {accuracy:.4f}')


def group_based(model, subjects, data_eda, time_sequences_feature_vectors, labels, classes):
    groups = get_groups() # TODO:implement

    for group in groups:
        for target_subject in group:
            for subject in group:
                if subject == target_subject:
                    continue


def main():
    print(f'Using {"cuda" if torch.cuda.is_available() else "cpu"} device')

    classes = (1, 4)
    sample_duration = round(8, 1)
    do_preprocessing = True
    batch_size = 40
    do_loso_run = True
    do_grid_search = False
    learning_rates = {'rnn': 0.0001, 'crnn': 0.05, 'feature_rnn': 0.05, 'cnn': 0.005}
    num_epochs = {'rnn': 100, 'crnn': 100, 'feature_rnn': 150, 'cnn': 100}
    hidden_state_sizes = {'rnn': 64, 'crnn': 64, 'feature_rnn': 64}
    num_recurrent_layers = {'rnn': 2, 'crnn': 2, 'feature_rnn': 2}
    use_grus = True
    # TODO: implement verbose output of deep models or not
    # TODO: implement option for individual confusion matrices
    # TODO: implement option for overall confusion matrices
    # TODO: decide what to do with training plots
    # TODO: add list or dict for all deep model options?

    subjects = save_load_data.get_subjects()
    
    data_eda, feature_vectors_eda, time_sequences_feature_vectors, labels = get_data(subjects, do_preprocessing, classes, sample_duration)

    print(data_eda.shape)
    
    if do_loso_run:
        loso(data_eda, feature_vectors_eda, time_sequences_feature_vectors, labels, subjects, classes, learning_rates, num_epochs, batch_size, hidden_state_sizes, num_recurrent_layers, use_grus)

    if do_grid_search:
        grid_search(data_eda, time_sequences_feature_vectors, labels, classes)

if __name__ == '__main__':
    # TODO: make .txt file of dependencies and order dependencies correctly
    # TODO: write documentation
    # TODO: implement grid search for hyperparameters
    # TODO: implement CLI
    main()