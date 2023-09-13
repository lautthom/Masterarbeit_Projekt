import preprocess_data
import save_load_data
import numpy as np
import baseline_models
import rnn_model
import crnn_model
import feature_rnn_model


def get_data(subjects, preprocess, compute_feature_vectors):
    if preprocess:
        # TODO: add options for classes, time length, and possible other options
        print('Getting data and cutting out samples...')
        
        data_eda, labels = preprocess_data.get_cut_out_samples_and_labels(subjects)
        save_load_data.save_samples(data_eda)
        save_load_data.save_labels(labels)
    else:
        print('Loading data...')
        data_eda = save_load_data.load_samples()
        labels = save_load_data.load_labels()
    # TODO: clean EDA data

    # TODO: create separate scripts for file management and manipulation of data
    # TODO: add all (correct) feature for feature vectors
    # TODO: check if feature vectors can be calculated faster (multithreading, applying functions to arrays without for-loop, etc.)
    if compute_feature_vectors:
        print('Computing feature vectors...')
        feature_vectors_eda = preprocess_data.compute_feature_vectors(data_eda)
        save_load_data.save_feature_vectors(feature_vectors_eda)
        time_sequences_feature_vectors = preprocess_data.compute_time_sequences_feature_vectors(data_eda)
        save_load_data.save_time_sequences_feature_vectors(time_sequences_feature_vectors)
    else:
        print('Loading feature vectors...')
        feature_vectors_eda = save_load_data.load_feature_vectors()
        time_sequences_feature_vectors = save_load_data.load_time_sequences_feature_vectors()

    return data_eda, labels, feature_vectors_eda, time_sequences_feature_vectors


def main():
    classes = (1, 4)
    preprocess = False
    compute_feature_vectors = False
    
    subjects = save_load_data.get_subjects()
    
    data_eda, labels, feature_vectors_eda, time_sequences_feature_vectors = get_data(subjects, preprocess, compute_feature_vectors)
    
    random_forest_accuracies = []
    rnn_accuracies = []
    crnn_accuracies = []
    feature_rnn_accuracies = []

    # TODO: normalize data for deep learning?
 
    # TODO: perform same training/evaluation split for all deep learning models
        # TODO: separate file for deep learning preprocessing?
        # TODO: feature RNN should use same data cutting and feature calculation method as general preprocessing

    # TODO: add option to save/load models
    # TODO: add options for batch size, learning rate, etc.

    # TODO: implement CRNN/RNN model in one file (possibly also feature RNN model)?
    
    # TODO: add CNN model?
    # TODO: add GRU unit for RNN models?

    # TODO: check if RNN model is correct

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
        data_training = data_training.reshape((86 * 40, 4096))
        feature_vectors_training = feature_vectors_training.reshape((86 * 40, 36))
        sequence_feature_vectors_training = sequence_feature_vectors_training.reshape((86 * 40, 8, 12))
        labels_training = labels_training.ravel()

        accuracy_forest = baseline_models.random_forest_model(feature_vectors_training, labels_training, feature_vectors_test, labels_test)
        print(f'Random Forest Accuracy: {accuracy_forest}')
        random_forest_accuracies.append(accuracy_forest)

        accuracy_rnn = rnn_model.run_model(data_training, labels_training, data_test, labels_test)
        print(f'RNN Model Accuracy: {accuracy_rnn}')
        rnn_accuracies.append(accuracy_rnn)

        accuracy_crnn = crnn_model.run_model(data_training, labels_training, data_test, labels_test)
        print(f'CRNN Model Accuracy: {accuracy_crnn}')
        crnn_accuracies.append(accuracy_crnn)

        accuracy_feature_rnn = feature_rnn_model.run_model(sequence_feature_vectors_training, labels_training, sequence_feature_vectors_test, labels_test)
        print(f'Feature RNN Model Accuracy: {accuracy_feature_rnn}')
        feature_rnn_accuracies.append(accuracy_feature_rnn)
        print(' ')
    
    # TODO: implement confusion matrix for results
    print(f'Mean Random Forest model Accuracy: {sum(random_forest_accuracies) / len(random_forest_accuracies):.4f}')
    print(f'Mean RNN model Accuracy: {sum(rnn_accuracies) / len(rnn_accuracies):.4f}')
    print(f'Mean CRNN model Accuracy: {sum(crnn_accuracies) / len(crnn_accuracies):.4f}')
    print(f'Mean feature RNN model Accuracy: {sum(feature_rnn_accuracies) / len(feature_rnn_accuracies):.4f}')

    for accuracy in rnn_accuracies:
        print(accuracy)
        


if __name__ == '__main__':
    # TODO: make .txt file of dependencies and order dependencies correctly
    # TODO: write documentation
    # TODO: implement grid search for hyperparameters
    # TODO: implement CLI
    main()