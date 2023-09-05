import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def make_subdirectories(subjects):
    for subject in subjects:
        try:
            os.makedirs(f'/home/lautthom/Desktop/processed_data/{subject}/training_data')
        except FileExistsError as error:
            print(f'Warning: {error}')
        try:
            os.mkdir(f'/home/lautthom/Desktop/processed_data/{subject}/test_data')
        except FileExistsError as error:
            print(f'Warning: {error}')

def save_data_proband(data, subjects, current_subject):
    for subject in subjects:
        if subject == current_subject:
            with open(f'/home/lautthom/Desktop/processed_data/{current_subject}/test_data/{subject}.npy', 'wb') as file:
                np.save(file, data)
        else:
            with open(f'/home/lautthom/Desktop/processed_data/{current_subject}/training_data/{subject}.npy', 'wb') as file:
                np.save(file, data)
            

def make_graphic(time_after_stimulus, subject, time_in_s):
    counter_0 = 0
    counter_4 = 0

    fig, axs = plt.subplots(5,8)
    fig.suptitle(f'GSR of T1 and T4 for {time_in_s}s after stimuli; Subject {subject}', fontsize=50)

    for item in time_after_stimulus:
        if item[0][2] == 1:
            axs[counter_0 // 4, counter_0 % 4].plot(item[:,0], item[:,1])
            counter_0 += 1
        if item[0][2] == 4:
            axs[counter_4 // 4, counter_4 % 4 + 4].plot(item[:,0], item[:,1])
            counter_4 += 1

    axs[0,1].set_title('T1 stimuli', x=1.05, y=1.05, fontsize=60)
    axs[0,5].set_title('T4 stimuli', x=1.05, y=1.05, fontsize=60)

    plt.show()


subjects_df = pd.read_csv('/home/lautthom/Desktop/PartC-Biosignals/samples.csv', sep='\t')
subjects = subjects_df.subject_name.tolist()

make_subdirectories(subjects)

lengths_between_stimuli = []
lengths_with_no_stimuli = []

for subject in subjects:
    data = pd.read_csv(f'/home/lautthom/Desktop/PartC-Biosignals/biosignals_raw/{subject}.csv', sep='\t')
    stimulus_data = pd.read_csv(f'/home/lautthom/Desktop/PartC-Biosignals/stimulus/{subject}.csv', sep='\t')
    temperature_data = pd.read_csv(f'/home/lautthom/Desktop/PartC-Biosignals/temperature/{subject}.csv', sep='\t')

    data_filtered = data.filter(items=['time', 'gsr'])

    # possible refactoring
    mask_label_1 = stimulus_data['label'] == 1 
    mask_label_4 = stimulus_data['label'] == 4
    total_mask = mask_label_1 | mask_label_4

    stimulus_data_labels = stimulus_data.loc[total_mask]

    merged_data_stimulus = pd.merge(data_filtered, stimulus_data_labels, on='time', how='left')
    #merged_data_stimulus = pd.merge(data_filtered, stimulus_data, on='time', how='left')

    merged_data_total = pd.merge(merged_data_stimulus, temperature_data, on='time', how='left')

    merged_array = merged_data_total.to_numpy()

    time_in_s = 8
    number_of_samples = time_in_s * 512

    datapoints_per_sample = merged_array[0].shape[0]

    time_after_stimulus = np.empty([0, number_of_samples, datapoints_per_sample])

    counter_between_stimuli = 0
    counter_no_stimuli = 0
    start_of_stimuli = False
    was_cooled_down = False
    start_of_measurement = False

    for index, item in enumerate(merged_array):
        if not np.isnan(item[2]):
            time_after_stimulus = np.append(time_after_stimulus, np.array([merged_array[index:index+number_of_samples]]), axis=0)
            start_of_stimuli = True
            was_cooled_down = False
            start_of_measurement = True
        if not np.isnan(item[3]) and item[3] < 33 and start_of_measurement:
            was_cooled_down = True
            counter_no_stimuli += 1
        elif was_cooled_down:
            counter_no_stimuli += 1
        if start_of_stimuli:
            counter_between_stimuli += 1
        if not np.isnan(item[3]) and item[3] > 33 and was_cooled_down:
            lengths_between_stimuli.append(counter_between_stimuli)
            lengths_with_no_stimuli.append(counter_no_stimuli)
            counter_between_stimuli = 0
            counter_no_stimuli = 0
            start_of_stimuli = False
            was_cooled_down = False
    lengths_between_stimuli.append(counter_between_stimuli)
    lengths_with_no_stimuli.append(counter_no_stimuli)

    print(time_after_stimulus.shape)

    save_data_proband(time_after_stimulus, subjects, subject)

print(f'No stimuli, min length: {min(lengths_with_no_stimuli) / 512}s')
print(f'With stimuli, min length: {min(lengths_between_stimuli) / 512}s')


