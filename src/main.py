import preprocess_data
import save_load_data
import numpy as np
import baseline_models
import sklearn
import deep_model
import torch
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import deep_learning_utils
import pathlib


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

 
def get_part_a_data(subjects, classes):

    data_eda, labels = load_part_a_samples(subjects, classes)
    #### data_eda_normalized doesn't work!!!
    data_eda_normalized = data_eda.copy()
    data_eda_normalized = data_eda_normalized / np.max(data_eda_normalized)
    data_eda_signal_tonic_phasic = preprocess_data.compute_tonic_and_phasic_components(data_eda)
    data_eda_signal_tonic_phasic_normalized = preprocess_data.compute_tonic_and_phasic_components(data_eda_normalized)

    print('Computing feature vectors...')
    feature_vectors_eda = preprocess_data.compute_feature_vectors(data_eda_signal_tonic_phasic)
    time_sequences_feature_vectors = preprocess_data.compute_time_sequences_feature_vectors(data_eda_signal_tonic_phasic)
    
    data_eda = preprocess_data.reduce_eda_signal(data_eda_signal_tonic_phasic)

    #### data_eda_normalized doesn't work!!!
    return data_eda, feature_vectors_eda, time_sequences_feature_vectors, labels


def load_part_a_samples(subjects, classes):
    
    samples = np.empty([0, 40, 1152])
    labels = np.empty([0, 40])

    for subject in subjects:
    

        data = np.load(pathlib.Path(f'/home/lautthom/Desktop/bioviddatasetfiles-master/bioviddatasetfiles-master/PartA/Test/GSR/12{subject[:4]}-{subject[4:]}_data.npy'))
        label = np.load(pathlib.Path(f'/home/lautthom/Desktop/bioviddatasetfiles-master/bioviddatasetfiles-master/PartA/Test/GSR/12{subject[:4]}-{subject[4:]}_label.npy'))

        label = np.ravel(label)

        mask_0 = label == 0
        mask_4 = label == 4
        total_mask = mask_0 | mask_4

        data = data[total_mask]
        label = label[total_mask]

        data = np.squeeze(data)

        data = np.expand_dims(data, axis=0)
        labels_proband = np.expand_dims(label, axis=0)

        samples = np.append(samples, data, axis=0)
        labels = np.append(labels, labels_proband, axis=0)

    return samples, labels



def get_group_indexes(subjects):
    groups = [[], [], [], [], [], []]
    for index, subject in enumerate(subjects):
        if subject[-4] == 'm' and int(subject[-2:]) < 36:
            groups[0].append(index)
        elif subject[-4] == 'm' and int(subject[-2:]) < 51:
            groups[1].append(index)
        elif subject[-4] == 'm':
            groups[2].append(index)
        elif subject[-4] == 'w' and int(subject[-2:]) < 36:
            groups[3].append(index)
        elif subject[-4] == 'w' and int(subject[-2:]) < 51:
            groups[4].append(index)
        elif subject[-4] == 'w':
            groups[5].append(index)
    return groups


def get_group_dataset(group, data, data_vectors, time_vectors, labels):
    dataset = np.empty([0, 40, 128, 3])
    vectors_dataset = np.empty([0, 40, 36])
    time_vectors_dataset = np.empty([0, 40, 8, 36])
    labels_group = np.empty([0, 40])
    for index in group:
        dataset = np.append(dataset, np.expand_dims(data[index], axis=0), axis=0)
        vectors_dataset = np.append(vectors_dataset, np.expand_dims(data_vectors[index], axis=0), axis=0)
        time_vectors_dataset = np.append(time_vectors_dataset, np.expand_dims(time_vectors[index], axis=0), axis=0)
        labels_group = np.append(labels_group, np.expand_dims(labels[index], axis=0), axis=0)
    return dataset, vectors_dataset, time_vectors_dataset, labels_group

def get_group_dataset_a(group, data, data_vectors, time_vectors, labels):
    dataset = np.empty([0, 40, 36, 3])
    vectors_dataset = np.empty([0, 40, 36])
    time_vectors_dataset = np.empty([0, 40, 2, 36])
    labels_group = np.empty([0, 40])
    for index in group:
        dataset = np.append(dataset, np.expand_dims(data[index], axis=0), axis=0)
        vectors_dataset = np.append(vectors_dataset, np.expand_dims(data_vectors[index], axis=0), axis=0)
        time_vectors_dataset = np.append(time_vectors_dataset, np.expand_dims(time_vectors[index], axis=0), axis=0)
        labels_group = np.append(labels_group, np.expand_dims(labels[index], axis=0), axis=0)
    return dataset, vectors_dataset, time_vectors_dataset, labels_group


def loso(data_eda, feature_vectors_eda, time_sequences_feature_vectors, labels, subjects, classes, learning_rates, num_epochs, batch_size, hidden_state_sizes, num_recurrent_layers, use_grus):
    random_forest_accuracies = []
    rnn_accuracies = []
    crnn_accuracies = []
    feature_rnn_accuracies = []
    cnn_accuracies = []

    rnn_labels = []
    crnn_labels = []
    feature_rnn_labels = []
    cnn_labels = []

    rnn_predictions = []
    crnn_predictions = []
    feature_rnn_predictions = []
    cnn_predictions = []

    genders = []
    age_groups = []

    # TODO: add option to save/load models?
    for index, (proband_data, proband_feature_vectors, proband_sequence_feature_vectors, proband_labels, proband_name) in enumerate(zip(data_eda, feature_vectors_eda, time_sequences_feature_vectors, labels, subjects)):
        print(' ')
        genders.append(proband_name[-4])
        if int(proband_name[-2:]) < 36:
            age_group = 'young'
        elif int(proband_name[-2:]) < 51:
            age_group = 'middle'
        else:
            age_group = 'old'
        age_groups.append(age_group)

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
        accuracy_rnn, labels_rnn, predictions_rnn, _ = deep_model.run_model('rnn', data_training, labels_training, data_test, labels_test, classes, learning_rates['rnn'], 50, batch_size, hidden_state_sizes['rnn'], num_recurrent_layers['rnn'], use_grus=use_grus)
        print(f'RNN Model Accuracy: {accuracy_rnn}')
        rnn_accuracies.append(accuracy_rnn)
        rnn_labels.append(labels_rnn)
        rnn_predictions.append(predictions_rnn)
        print(' ')

        print('Running CRNN model...')
        accuracy_crnn, labels_crnn, predictions_crnn, _ = deep_model.run_model('crnn', data_training, labels_training, data_test, labels_test, classes, learning_rates['crnn'], 50, batch_size, hidden_state_sizes['crnn'], num_recurrent_layers['crnn'], use_grus=use_grus)
        print(f'CRNN Model Accuracy: {accuracy_crnn}')
        crnn_accuracies.append(accuracy_crnn)
        crnn_labels.append(labels_crnn)
        crnn_predictions.append(predictions_crnn)
        print(' ')

        print('Running Feature RNN model...')
        accuracy_feature_rnn, labels_feature_rnn, predictions_feature_rnn, _ = deep_model.run_model('feature_rnn', sequence_feature_vectors_training, labels_training, sequence_feature_vectors_test, labels_test, classes, learning_rates['feature_rnn'], 50, batch_size, hidden_state_sizes['feature_rnn'], num_recurrent_layers['rnn'], use_grus=use_grus)
        print(f'Feature RNN Model Accuracy: {accuracy_feature_rnn}')
        feature_rnn_accuracies.append(accuracy_feature_rnn)
        feature_rnn_labels.append(labels_feature_rnn)
        feature_rnn_predictions.append(predictions_feature_rnn)
        print(' ')

        print('Running CNN model...')
        accuracy_cnn, labels_cnn, predictions_cnn, _ = deep_model.run_model('cnn', data_training, labels_training, data_test, labels_test, classes, learning_rates['cnn'], 50, batch_size)
        print(f'CNN Model Accuracy: {accuracy_cnn}')
        cnn_accuracies.append(accuracy_cnn)
        cnn_labels.append(labels_cnn)
        cnn_predictions.append(predictions_cnn)
        print(' ')
    
    # TODO: implement confusion matrix for results
    print(f'Mean Random Forest model Accuracy: {sum(random_forest_accuracies) / len(random_forest_accuracies):.4f}')
    print(f'Mean RNN model Accuracy: {sum(rnn_accuracies) / len(rnn_accuracies):.4f}')
    print(f'Mean CRNN model Accuracy: {sum(crnn_accuracies) / len(crnn_accuracies):.4f}')
    print(f'Mean feature RNN model Accuracy: {sum(feature_rnn_accuracies) / len(feature_rnn_accuracies):.4f}')
    print(f'Mean CNN model Accuracy: {sum(cnn_accuracies) / len(cnn_accuracies):.4f}')

    results_df = pd.DataFrame(list(zip(subjects, genders, age_groups, random_forest_accuracies, rnn_accuracies, crnn_accuracies, feature_rnn_accuracies, cnn_accuracies)), columns=['Subjects', 'Accuracy Random Forest', 'Accuracy RNN', 'Accuracy CRNN', 'Accuracy Feature RNN', 'Accuracy CNN'])
    save_load_data.save_results(results_df)

    rnn_labels = np.array(rnn_labels).ravel()
    rnn_predictions = np.array(rnn_predictions).ravel()
    crnn_labels = np.array(crnn_labels).ravel()
    crnn_predictions = np.array(crnn_predictions).ravel()
    feature_rnn_labels = np.array(feature_rnn_labels).ravel()
    feature_rnn_predictions = np.array(feature_rnn_predictions).ravel()
    cnn_labels = np.array(cnn_labels).ravel()
    cnn_predictions = np.array(cnn_predictions).ravel() 

    ConfusionMatrixDisplay.from_predictions(rnn_labels, rnn_predictions)
    plt.title('Results RNN')
    plt.show()
    ConfusionMatrixDisplay.from_predictions(crnn_labels, crnn_predictions)
    plt.title('Results CRNN')
    plt.show()
    ConfusionMatrixDisplay.from_predictions(feature_rnn_labels, feature_rnn_predictions)
    plt.title('Results Feature RNN')
    plt.show()
    ConfusionMatrixDisplay.from_predictions(cnn_labels, cnn_predictions)
    plt.title('Results CNN')
    plt.show()


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
                accuracy, _, _, _ = deep_model.run_model('cnn', data_train, labels_train, data_test, labels_test, classes, learning_rate, epoch, 40, kernel_size=kernel_size)
                print(f'For CNN model using kernel size {kernel_size}, {learning_rate} learning rate, {epoch} epochs: accuracy {accuracy:.4f}')


def group_based(model, subjects, data_eda, time_sequences_feature_vectors, labels, classes, epochs, finetuning=False):
    total_accuracy = []
    subject_list = []

    groups = get_group_indexes(subjects) 

    for full_indexes in groups:
        group_accuracy = []

        group_dataset, group_vectors, group_labels = get_group_dataset(full_indexes, data_eda, time_sequences_feature_vectors, labels)

        group_indexes = [i for i in range(len(full_indexes))]

        for full_index, group_index in zip(full_indexes, group_indexes):
            data_test = data_eda[full_index]
            if model == 'feature_rnn':
                data_test = time_sequences_feature_vectors[full_index]
            labels_test = labels[full_index]

            if not finetuning:
                data_training = np.delete(group_dataset, np.s_[group_index], axis=0)
                labels_training = np.delete(group_labels, np.s_[group_index], 0)
                if model == 'feature_rnn':
                    data_training = np.delete(group_vectors, np.s_[group_index], 0)
            else:
                data_training = np.delete(data_eda, np.s_[full_index], 0)
                labels_training = np.delete(labels, np.s_[full_index], 0)
                if model == 'feature_rnn':
                    data_training = np.delete(time_sequences_feature_vectors, np.s_[full_index], 0)

            data_finetuning = np.delete(group_dataset, np.s_[group_index], 0)
            labels_finetuning = np.delete(group_labels, np.s_[group_index], 0)
            if model == 'feature_rnn':
                data_finetuning = np.delete(group_vectors, np.s_[group_index], 0)
            
            if not model == 'feature_rnn':
                data_training = data_training.reshape((data_training.shape[0] * 40, data_training.shape[2], 3))
                data_finetuning = data_finetuning.reshape((data_finetuning.shape[0] * 40, data_finetuning.shape[2], 3))
            else:
                data_training = data_training.reshape((data_training.shape[0] * 40, data_training.shape[2], 36))
                data_finetuning = data_finetuning.reshape((data_finetuning.shape[0] * 40, data_finetuning.shape[2], 36))
            
            labels_training = labels_training.ravel()
            labels_finetuning = labels_finetuning.ravel()

            # weglassen? entscheiden, wo geshuffled wird  
            data_training, labels_training = sklearn.utils.shuffle(data_training, labels_training)
            data_finetuning, labels_finetuning = sklearn.utils.shuffle(data_finetuning, labels_finetuning)
            data_test, labels_test = sklearn.utils.shuffle(data_test, labels_test)

            try:
               net = save_load_data.load_model('finetuning_100_rnn', full_index)
            except FileNotFoundError:
               accuracy_rnn, labels_rnn, predictions_rnn, net = deep_model.run_model(model, data_training, labels_training, data_training, labels_training, classes, 0, epochs, 40, hidden_state_size=64, num_recurrent_layer=2, use_grus=True, finetuning_data=data_finetuning, finetuning_labels=labels_finetuning)

            accuracy_rnn, labels_rnn, predictions_rnn, net = deep_model.run_model(model, data_training, labels_training, data_test, labels_test, classes, 0, epochs, 40, hidden_state_size=64, num_recurrent_layer=2, use_grus=True, finetuning_data=data_finetuning, finetuning_labels=labels_finetuning)

            # if model == 'crnn' or model == 'cnn':
            #     data_test = np.transpose(data_test, (0, 2, 1))

            # labels_test = labels_test.copy()
            # labels_test = deep_learning_utils.relabel(labels_test, classes)

            # test_dataloader = deep_learning_utils.make_dataloader(data_test, labels_test, 40, shuffle=False)

            # accuracy, _, _, outputs = deep_model.run_evaluation(net, test_dataloader, 'cuda')

            # save_load_data.save_model(net, 'finetuning_100_rnn', full_index)

            # predictions = []
            # labels_evaluation = []
            # for output, label in zip(outputs, labels_test):
                
            #     prediction = 1 if output >= 0.5 else 0
            #     predictions.append(prediction)
            #     labels_evaluation.append(label.item())

            group_accuracy.append(accuracy_rnn)
            total_accuracy.append(accuracy_rnn)
            subject_list.append(subjects[full_index])

            print(f'Target subject {subjects[full_index]} accuracy: {accuracy_rnn:.4f}')
        print(' ')
        print(f'Group accuracy: {sum(group_accuracy) / len(group_accuracy):.4f}')
    print(f'Total accuracy: {sum(total_accuracy) / len(total_accuracy):.4f}')

    results_df = pd.DataFrame(list(zip(subject_list, total_accuracy)), columns=['Subjects', 'Accuracy RNN'])
    filename = f'{"group" if not finetuning else "finetuning"}_{epochs}_{model}'
    results_df.to_csv(pathlib.Path(f'src/../results/{filename}.csv'), index=False)
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
                accuracy, _, _, _ = deep_model.run_model('cnn', data_train, labels_train, data_test, labels_test, classes, learning_rate, epoch, 40, kernel_size=kernel_size)
                print(f'For CNN model using kernel size {kernel_size}, {learning_rate} learning rate, {epoch} epochs: accuracy {accuracy:.4f}')


def do_loso_run(subjects, data_eda, feature_vectors, time_sequences_feature_vectors, labels, classes, group=False, finetuning=False, part_a=False):
    if group or finetuning:
        epochs = 100
    else:
        epochs = 50

    testing_mode = 'group' if group else 'finetuning' if finetuning else 'complete'
        
    print(f'Starting LOSO run with {testing_mode} datasets...\n')

    random_forest_accuracies = []
    cnn_accuracies = []
    rnn_accuracies = []
    crnn_accuracies = []
    feature_rnn_accuracies = []
    ensemble_accuracies = []

    ensemble_cnn_rnn = []
    ensemble_cnn_feature_rnn = []
    ensemble_cnn_crnn = []
    ensemble_rnn_feature_rnn = []
    ensemble_rnn_crnn = []
    ensemble_feature_rnn_crnn = []

    ensemble_cnn_rnn_feature_rnn = []
    ensemble_cnn_rnn_crnn = []
    ensemble_rnn_feature_rnn_crnn = []
    
    cnn_outputs = []
    rnn_outputs = []
    crnn_outputs = []
    feature_rnn_outputs = []  
    ensemble_outputs = []

    forest_predictions = []

    subject_list = []

    group_numbers = []

    accumulated_labels = []

    groups = get_group_indexes(subjects) 

    for group_number, full_indexes in enumerate(groups):

        if not part_a:
            group_dataset, group_vectors, group_time_vectors, group_labels = get_group_dataset(full_indexes, data_eda, feature_vectors, time_sequences_feature_vectors, labels)
        else:
            group_dataset, group_vectors, group_time_vectors, group_labels = get_group_dataset_a(full_indexes, data_eda, feature_vectors, time_sequences_feature_vectors, labels)
            
        group_indexes = [i for i in range(len(full_indexes))]

        for full_index, group_index in zip(full_indexes, group_indexes):
            print(f'Current subject: {subjects[full_index]}')
        
            data_test = data_eda[full_index]
            vectors_test = feature_vectors[full_index]
            time_sequences_vectors_test = time_sequences_feature_vectors[full_index]
            labels_test = labels[full_index]

            if group:
                data_training = np.delete(group_dataset, np.s_[group_index], axis=0)
                vectors_training = np.delete(group_vectors, np.s_[group_index], 0)                
                time_sequences_vectors_training = np.delete(group_time_vectors, np.s_[group_index], 0)
                labels_training = np.delete(group_labels, np.s_[group_index], 0)
            else:
                data_training = np.delete(data_eda, np.s_[full_index], 0)
                vectors_training = np.delete(feature_vectors, np.s_[full_index], 0)             
                time_sequences_vectors_training = np.delete(time_sequences_feature_vectors, np.s_[full_index], 0)
                labels_training = np.delete(labels, np.s_[full_index], 0)

            if finetuning or group:               
                data_finetuning = np.delete(group_dataset, np.s_[group_index], axis=0)              
                time_sequences_vectors_finetuning = np.delete(group_time_vectors, np.s_[group_index], 0)
                labels_finetuning = np.delete(group_labels, np.s_[group_index], 0)
            else:
                data_finetuning = np.delete(data_eda, np.s_[full_index], 0)        
                time_sequences_vectors_finetuning = np.delete(time_sequences_feature_vectors, np.s_[full_index], 0)
                labels_finetuning = np.delete(labels, np.s_[full_index], 0)
            
            data_training = data_training.reshape((data_training.shape[0] * 40, data_training.shape[2], 3))
            data_finetuning = data_finetuning.reshape((data_finetuning.shape[0] * 40, data_finetuning.shape[2], 3))
            vectors_training = vectors_training.reshape((vectors_training.shape[0] * 40, vectors_training.shape[2]))
            time_sequences_vectors_training = time_sequences_vectors_training.reshape((time_sequences_vectors_training.shape[0] * 40, time_sequences_vectors_training.shape[2], 36))
            time_sequences_vectors_finetuning = time_sequences_vectors_finetuning.reshape((time_sequences_vectors_finetuning.shape[0] * 40, time_sequences_vectors_finetuning.shape[2], 36))
            
            labels_training = labels_training.ravel()
            labels_finetuning = labels_finetuning.ravel()

            # weglassen? entscheiden, wo geshuffled wird  
            data_training, vectors_training, time_sequences_vectors_training, labels_training = sklearn.utils.shuffle(data_training, vectors_training, time_sequences_vectors_training, labels_training)
            data_finetuning, time_sequences_vectors_finetuning, labels_finetuning = sklearn.utils.shuffle(data_finetuning, time_sequences_vectors_finetuning, labels_finetuning)

            if not finetuning:
                try: 
                    model = save_load_data.load_forest(testing_mode, full_index)
                    predictions_forest = baseline_models.run_loaded_forest(model, vectors_test, labels_test)
                except FileNotFoundError:
                    print('Training Random Forest...')
                    predictions_forest, model = baseline_models.random_forest_model(vectors_training, labels_training, vectors_test)
                    save_load_data.save_forest(model, testing_mode, full_index)
                accuracy_forest = accuracy_score(labels_test, predictions_forest)
                print(f'Random Forest Accuracy: {accuracy_forest}')
                random_forest_accuracies.append(accuracy_forest)
                forest_predictions.extend(predictions_forest)

            #Make enum (or comparable) for type of model
            
            try:
                model = save_load_data.load_model(testing_mode, 'cnn', full_index)
                accuracy_cnn, outputs_cnn = deep_model.run_loaded_model(model, 'cnn', data_test, labels_test, classes)
            except FileNotFoundError:
                print('Training CNN model...')
                accuracy_cnn, outputs_cnn, model = deep_model.run_model('cnn', data_training, labels_training, data_test, labels_test, classes, 0, epochs, 40, finetuning_data=data_finetuning, finetuning_labels=labels_finetuning)
                save_load_data.save_model(model, testing_mode, 'cnn', full_index)
            print(f'CNN Model Accuracy: {accuracy_cnn}')
            cnn_accuracies.append(accuracy_cnn)
            cnn_outputs.extend(outputs_cnn)

            try:
                model = save_load_data.load_model(testing_mode, 'rnn', full_index)
                accuracy_rnn, outputs_rnn = deep_model.run_loaded_model(model, 'rnn', data_test, labels_test, classes)
            except FileNotFoundError:
                print('Training RNN model...')
                accuracy_rnn, outputs_rnn, model = deep_model.run_model('rnn', data_training, labels_training, data_test, labels_test, classes, 0, epochs, 40, 64, 2, use_grus=True, finetuning_data=data_finetuning, finetuning_labels=labels_finetuning)
                save_load_data.save_model(model, testing_mode, 'rnn', full_index)
            print(f'RNN Model Accuracy: {accuracy_rnn}')
            rnn_accuracies.append(accuracy_rnn)
            rnn_outputs.extend(outputs_rnn)
            
            try:
                model = save_load_data.load_model(testing_mode, 'crnn', full_index)
                accuracy_crnn, outputs_crnn = deep_model.run_loaded_model(model, 'crnn', data_test, labels_test, classes)
            except FileNotFoundError:
                print('Training CRNN model...')
                accuracy_crnn, outputs_crnn, model = deep_model.run_model('crnn', data_training, labels_training, data_test, labels_test, classes, 0, epochs, 40, 64, 2, use_grus=True, finetuning_data=data_finetuning, finetuning_labels=labels_finetuning)
                save_load_data.save_model(model, testing_mode, 'crnn', full_index)
            print(f'CRNN Model Accuracy: {accuracy_crnn}')
            crnn_accuracies.append(accuracy_crnn)
            crnn_outputs.extend(outputs_crnn)

            try:
                model = save_load_data.load_model(testing_mode, 'feature_rnn', full_index)
                accuracy_feature_rnn, outputs_feature_rnn = deep_model.run_loaded_model(model, 'feature_rnn', time_sequences_vectors_test, labels_test, classes)
            except FileNotFoundError:
                print('Training Feature RNN model...')
                accuracy_feature_rnn, outputs_feature_rnn, model = deep_model.run_model('feature_rnn', time_sequences_vectors_training, labels_training, time_sequences_vectors_test, labels_test, classes, 0, epochs, 40, 64, 2, use_grus=True, finetuning_data=time_sequences_vectors_finetuning, finetuning_labels=labels_finetuning)
                save_load_data.save_model(model, testing_mode, 'feature_rnn', full_index)
            print(f'Feature RNN Model Accuracy: {accuracy_feature_rnn}')
            feature_rnn_accuracies.append(accuracy_feature_rnn)
            feature_rnn_outputs.extend(outputs_feature_rnn)

            means_ensemble = np.mean(np.array([outputs_cnn, outputs_rnn, outputs_crnn, outputs_feature_rnn]), axis=0)
            predictions_ensemble = np.where(means_ensemble < 0.5, 1, 4)
            accuracy_ensemble = accuracy_score(labels_test, predictions_ensemble)
            print(f'Ensemble accuracy: {accuracy_score(labels_test, predictions_ensemble)}\n')

            ensemble_outputs.extend(means_ensemble)
            ensemble_accuracies.append(accuracy_ensemble)



            means_ensemble = np.mean(np.array([outputs_cnn, outputs_rnn]), axis=0)
            predictions_ensemble = np.where(means_ensemble < 0.5, 1, 4)
            accuracy_ensemble = accuracy_score(labels_test, predictions_ensemble)

            ensemble_cnn_rnn.append(accuracy_ensemble)

            means_ensemble = np.mean(np.array([outputs_cnn, outputs_crnn]), axis=0)
            predictions_ensemble = np.where(means_ensemble < 0.5, 1, 4)
            accuracy_ensemble = accuracy_score(labels_test, predictions_ensemble)

            ensemble_cnn_crnn.append(accuracy_ensemble)

            means_ensemble = np.mean(np.array([outputs_cnn, outputs_feature_rnn]), axis=0)
            predictions_ensemble = np.where(means_ensemble < 0.5, 1, 4)
            accuracy_ensemble = accuracy_score(labels_test, predictions_ensemble)

            ensemble_cnn_feature_rnn.append(accuracy_ensemble)

            means_ensemble = np.mean(np.array([outputs_rnn, outputs_feature_rnn]), axis=0)
            predictions_ensemble = np.where(means_ensemble < 0.5, 1, 4)
            accuracy_ensemble = accuracy_score(labels_test, predictions_ensemble)

            ensemble_rnn_feature_rnn.append(accuracy_ensemble)

            means_ensemble = np.mean(np.array([outputs_rnn, outputs_crnn]), axis=0)
            predictions_ensemble = np.where(means_ensemble < 0.5, 1, 4)
            accuracy_ensemble = accuracy_score(labels_test, predictions_ensemble)

            ensemble_rnn_crnn.append(accuracy_ensemble)

            means_ensemble = np.mean(np.array([outputs_crnn, outputs_feature_rnn]), axis=0)
            predictions_ensemble = np.where(means_ensemble < 0.5, 1, 4)
            accuracy_ensemble = accuracy_score(labels_test, predictions_ensemble)

            ensemble_feature_rnn_crnn.append(accuracy_ensemble)

            means_ensemble = np.mean(np.array([outputs_cnn, outputs_rnn, outputs_feature_rnn]), axis=0)
            predictions_ensemble = np.where(means_ensemble < 0.5, 1, 4)
            accuracy_ensemble = accuracy_score(labels_test, predictions_ensemble)

            ensemble_cnn_rnn_feature_rnn.append(accuracy_ensemble)

            means_ensemble = np.mean(np.array([outputs_cnn, outputs_rnn, outputs_crnn]), axis=0)
            predictions_ensemble = np.where(means_ensemble < 0.5, 1, 4)
            accuracy_ensemble = accuracy_score(labels_test, predictions_ensemble)

            ensemble_cnn_rnn_crnn.append(accuracy_ensemble)

            means_ensemble = np.mean(np.array([outputs_rnn, outputs_feature_rnn, outputs_crnn]), axis=0)
            predictions_ensemble = np.where(means_ensemble < 0.5, 1, 4)
            accuracy_ensemble = accuracy_score(labels_test, predictions_ensemble)

            ensemble_rnn_feature_rnn_crnn.append(accuracy_ensemble)





            subject_list.append(subjects[full_index])
            group_numbers.append(group_number)

            accumulated_labels.extend(list(labels_test))

    if not finetuning:            
        results_df = pd.DataFrame(list(zip(subject_list, group_numbers, random_forest_accuracies, cnn_accuracies, rnn_accuracies, crnn_accuracies, feature_rnn_accuracies, ensemble_accuracies)), columns=['Subject', 'Group', 'Accuracy Random Forest', 'Accuracy CNN', 'Accuracy RNN', 'Accuracy CRNN', 'Accuracy Feature RNN', 'Accuracy Ensemble'])
    else:
        results_df = pd.DataFrame(list(zip(subject_list, group_numbers, cnn_accuracies, rnn_accuracies, crnn_accuracies, feature_rnn_accuracies, ensemble_accuracies)), columns=['Subject', 'Group', 'Accuracy CNN', 'Accuracy RNN', 'Accuracy CRNN', 'Accuracy Feature RNN', 'Accuracy Ensemble'])
    save_load_data.save_results(results_df, testing_mode)

    print('Overall results:')
    if not finetuning:
        print(f'Mean Random Forest model Accuracy: {sum(random_forest_accuracies) / len(random_forest_accuracies):.4f}')
    print(f'Mean CNN model Accuracy: {sum(cnn_accuracies) / len(cnn_accuracies):.4f}')
    print(f'Mean RNN model Accuracy: {sum(rnn_accuracies) / len(rnn_accuracies):.4f}')
    print(f'Mean CRNN model Accuracy: {sum(crnn_accuracies) / len(crnn_accuracies):.4f}')
    print(f'Mean feature RNN model Accuracy: {sum(feature_rnn_accuracies) / len(feature_rnn_accuracies):.4f}')

    print(f'Mean Ensemble accuracy: {sum(ensemble_accuracies) / len(ensemble_accuracies):.4f}')

    print(' ')
    print(f'Mean CNN RNN: {sum(ensemble_cnn_rnn) / len(ensemble_cnn_rnn):.4f}')
    print(f'Mean CNN feature RNN: {sum(ensemble_cnn_feature_rnn) / len(ensemble_cnn_feature_rnn):.4f}')
    print(f'Mean CNN CRNN: {sum(ensemble_cnn_crnn) / len(ensemble_cnn_crnn):.4f}')
    print(f'Mean RNN feature RNN: {sum(ensemble_rnn_feature_rnn) / len(ensemble_rnn_feature_rnn):.4f}')
    print(f'Mean RNN CRNN: {sum(ensemble_rnn_crnn) / len(ensemble_rnn_crnn):.4f}')
    print(f'Mean feature RNN CRNN: {sum(ensemble_feature_rnn_crnn) / len(ensemble_feature_rnn_crnn):.4f}')
    print(' ')

    print(f'Mean CNN RNN feature RNN: {sum(ensemble_cnn_rnn_feature_rnn) / len(ensemble_cnn_rnn_feature_rnn):.4f}')
    print(f'Mean CNN RNN CRNN: {sum(ensemble_cnn_rnn_crnn) / len(ensemble_cnn_rnn_crnn):.4f}')
    print(f'Mean RNN feature RNN CRNN: {sum(ensemble_rnn_feature_rnn_crnn) / len(ensemble_rnn_feature_rnn_crnn):.4f}')

    cnn_outputs_adjusted = np.array(cnn_outputs) * 4 + 1    
    rnn_outputs_adjusted = np.array(rnn_outputs) * 4 + 1
    crnn_outputs_adjusted = np.array(crnn_outputs) * 4 + 1
    feature_rnn_outputs_adjusted = np.array(feature_rnn_outputs) * 4 + 1
    ensemble_outputs_adjusted = np.array(ensemble_outputs) * 4 + 1

    auc_cnn = roc_auc_score(accumulated_labels, cnn_outputs_adjusted)
    auc_rnn = roc_auc_score(accumulated_labels, rnn_outputs_adjusted)
    auc_crnn = roc_auc_score(accumulated_labels, crnn_outputs_adjusted)
    auc_feature_rnn = roc_auc_score(accumulated_labels, feature_rnn_outputs_adjusted)
    auc_ensemble = roc_auc_score(accumulated_labels, ensemble_outputs_adjusted)

    print(' ')
    print(f'Mean CNN model AUC: {auc_cnn:.4f}')
    print(f'Mean RNN model AUC: {auc_rnn:.4f}')
    print(f'Mean CRNN model AUC: {auc_crnn:.4f}')
    print(f'Mean feature RNN model AUC: {auc_feature_rnn:.4f}')
    print(f'Ensemble AUC: {auc_ensemble:.4f}')
    
    if not finetuning:
        ConfusionMatrixDisplay.from_predictions(accumulated_labels, forest_predictions)
        plt.title('Confusion Matrix Random Forest')
        plt.show()
    ConfusionMatrixDisplay.from_predictions(accumulated_labels, np.where(np.array(cnn_outputs) < 0.5, 1, 4))
    plt.title('Results CNN')
    plt.show()
    ConfusionMatrixDisplay.from_predictions(accumulated_labels, np.where(np.array(rnn_outputs) < 0.5, 1, 4))
    plt.title('Results RNN')
    plt.show()
    ConfusionMatrixDisplay.from_predictions(accumulated_labels, np.where(np.array(crnn_outputs) < 0.5, 1, 4))
    plt.title('Results CRNN')
    plt.show()
    ConfusionMatrixDisplay.from_predictions(accumulated_labels, np.where(np.array(feature_rnn_outputs) < 0.5, 1, 4))
    plt.title('Results Feature RNN')
    plt.show()
    ConfusionMatrixDisplay.from_predictions(accumulated_labels, np.where(np.array(ensemble_outputs) < 0.5, 1, 4))
    plt.title('Results Ensemble')
    plt.show()
    
    RocCurveDisplay.from_predictions(accumulated_labels, cnn_outputs_adjusted, name='CNN ROC curve', pos_label=4, plot_chance_level=True)
    plt.show()
    RocCurveDisplay.from_predictions(accumulated_labels, rnn_outputs_adjusted, pos_label=4, plot_chance_level=True)
    plt.show()
    RocCurveDisplay.from_predictions(accumulated_labels, crnn_outputs_adjusted, pos_label=4, plot_chance_level=True)
    plt.show()
    RocCurveDisplay.from_predictions(accumulated_labels, feature_rnn_outputs_adjusted, pos_label=4, plot_chance_level=True)
    plt.show()
    RocCurveDisplay.from_predictions(accumulated_labels, ensemble_outputs_adjusted, pos_label=4, plot_chance_level=True)
    plt.show()










def main():
    print(f'Using {"cuda" if torch.cuda.is_available() else "cpu"} device')

    classes = (1, 4)
    sample_duration = round(8, 1)
    do_preprocessing = False
    batch_size = 40
    do_loso_run_flag = False
    do_grid_search = False
    do_group_based = False
    learning_rates = {'rnn': 0.0001, 'crnn': 0.05, 'feature_rnn': 0.05, 'cnn': 0.005}
    num_epochs = {'rnn': 100, 'crnn': 100, 'feature_rnn': 150, 'cnn': 100}
    hidden_state_sizes = {'rnn': 64, 'crnn': 64, 'feature_rnn': 64}
    num_recurrent_layers = {'rnn': 2, 'crnn': 2, 'feature_rnn': 2}
    use_grus = True

    part_a = True
    group = False
    finetuning = True
    epochs = 100
    # TODO: implement verbose output of deep models or not
    # TODO: implement option for individual confusion matrices
    # TODO: implement option for overall confusion matrices
    # TODO: decide what to do with training plots
    # TODO: add list or dict for all deep model options?

    subjects = save_load_data.get_subjects()

    if not part_a:    
        data_eda, feature_vectors_eda, time_sequences_feature_vectors, labels = get_data(subjects, do_preprocessing, classes, sample_duration)
    else:
        data_eda, feature_vectors_eda, time_sequences_feature_vectors, labels = get_part_a_data(subjects, classes)
    
    if do_loso_run_flag:
        loso(data_eda, feature_vectors_eda, time_sequences_feature_vectors, labels, subjects, classes, learning_rates, num_epochs, batch_size, hidden_state_sizes, num_recurrent_layers, use_grus)

    if do_grid_search:
        grid_search(data_eda, time_sequences_feature_vectors, labels, classes)

    if do_group_based:
        group_based('feature_rnn', subjects, data_eda, time_sequences_feature_vectors, labels, classes, epochs, finetuning=finetuning)

    do_loso_run(subjects, data_eda, feature_vectors_eda, time_sequences_feature_vectors, labels, classes, group=group, finetuning=finetuning, part_a=part_a)

if __name__ == '__main__':
    # TODO: make .txt file of dependencies and order dependencies correctly
    # TODO: write documentation
    # TODO: implement CLI
    main()