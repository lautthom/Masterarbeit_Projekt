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


def get_data(subjects, do_preprocessing, classes, sample_duration):
    if do_preprocessing:
        print('Getting data and cutting out samples...')
        data_eda, labels = preprocess_data.get_cut_out_samples_and_labels(subjects, classes, sample_duration)
        data_eda_signal_tonic_phasic = preprocess_data.compute_tonic_and_phasic_components(data_eda)

        print('Computing feature vectors...')
        feature_vectors_eda = preprocess_data.compute_feature_vectors(data_eda_signal_tonic_phasic)
        time_sequences_feature_vectors = preprocess_data.compute_time_sequences_feature_vectors(data_eda_signal_tonic_phasic)
       
        data_eda = preprocess_data.reduce_eda_signal(data_eda_signal_tonic_phasic)
        
        save_load_data.save_samples(data_eda, sample_duration, classes)
        save_load_data.save_labels(labels, classes)   
        save_load_data.save_feature_vectors(feature_vectors_eda, sample_duration, classes)
        save_load_data.save_time_sequences_feature_vectors(time_sequences_feature_vectors, sample_duration, classes)
        
    else:
        print('Loading data and feature vectors...')
        data_eda = save_load_data.load_samples(sample_duration, classes)
        labels = save_load_data.load_labels(classes)
        feature_vectors_eda = save_load_data.load_feature_vectors(sample_duration, classes)
        time_sequences_feature_vectors = save_load_data.load_time_sequences_feature_vectors(sample_duration, classes)

    return data_eda, feature_vectors_eda, time_sequences_feature_vectors, labels


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


def do_loso_run(subjects, data_eda, feature_vectors, time_sequences_feature_vectors, labels, classes, group=False, finetuning=False):
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

        group_dataset, group_vectors, group_time_vectors, group_labels = get_group_dataset(full_indexes, data_eda, feature_vectors, time_sequences_feature_vectors, labels)
            
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
                accuracy_cnn, outputs_cnn, model = deep_model.run_model('cnn', data_training, labels_training, data_test, labels_test, classes, epochs, 40, finetuning_data=data_finetuning, finetuning_labels=labels_finetuning)
                save_load_data.save_model(model, testing_mode, 'cnn', full_index)
            print(f'CNN Model Accuracy: {accuracy_cnn}')
            cnn_accuracies.append(accuracy_cnn)
            cnn_outputs.extend(outputs_cnn)

            try:
                model = save_load_data.load_model(testing_mode, 'rnn', full_index)
                accuracy_rnn, outputs_rnn = deep_model.run_loaded_model(model, 'rnn', data_test, labels_test, classes)
            except FileNotFoundError:
                print('Training RNN model...')
                accuracy_rnn, outputs_rnn, model = deep_model.run_model('rnn', data_training, labels_training, data_test, labels_test, classes, epochs, 40, 64, 2, use_grus=True, finetuning_data=data_finetuning, finetuning_labels=labels_finetuning)
                save_load_data.save_model(model, testing_mode, 'rnn', full_index)
            print(f'RNN Model Accuracy: {accuracy_rnn}')
            rnn_accuracies.append(accuracy_rnn)
            rnn_outputs.extend(outputs_rnn)
            
            try:
                model = save_load_data.load_model(testing_mode, 'crnn', full_index)
                accuracy_crnn, outputs_crnn = deep_model.run_loaded_model(model, 'crnn', data_test, labels_test, classes)
            except FileNotFoundError:
                print('Training CRNN model...')
                accuracy_crnn, outputs_crnn, model = deep_model.run_model('crnn', data_training, labels_training, data_test, labels_test, classes, epochs, 40, 64, 2, use_grus=True, finetuning_data=data_finetuning, finetuning_labels=labels_finetuning)
                save_load_data.save_model(model, testing_mode, 'crnn', full_index)
            print(f'CRNN Model Accuracy: {accuracy_crnn}')
            crnn_accuracies.append(accuracy_crnn)
            crnn_outputs.extend(outputs_crnn)

            try:
                model = save_load_data.load_model(testing_mode, 'feature_rnn', full_index)
                accuracy_feature_rnn, outputs_feature_rnn = deep_model.run_loaded_model(model, 'feature_rnn', time_sequences_vectors_test, labels_test, classes)
            except FileNotFoundError:
                print('Training Feature RNN model...')
                accuracy_feature_rnn, outputs_feature_rnn, model = deep_model.run_model('feature_rnn', time_sequences_vectors_training, labels_training, time_sequences_vectors_test, labels_test, classes, epochs, 40, 64, 2, use_grus=True, finetuning_data=time_sequences_vectors_finetuning, finetuning_labels=labels_finetuning)
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

    do_preprocessing = True
    group = False
    finetuning = True

    subjects = save_load_data.get_subjects()

    data_eda, feature_vectors_eda, time_sequences_feature_vectors, labels = get_data(subjects, do_preprocessing, classes, sample_duration)

    do_loso_run(subjects, data_eda, feature_vectors_eda, time_sequences_feature_vectors, labels, classes, group=group, finetuning=finetuning)

if __name__ == '__main__':
    main()