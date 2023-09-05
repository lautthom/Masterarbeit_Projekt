import tensorflow as tf
import numpy as np

def relabel(label_array):
    label_array[label_array == 1] = 0
    label_array[label_array == 4] = 1
    return label_array


def run_model(data_training, labels_train, data_test, labels_test):
    print(tf.config.list_physical_devices('GPU'))

    labels_train = relabel(labels_train)
    labels_test = relabel(labels_test)

    model = tf.keras.Sequential()
    #model.add(tf.keras.layers.Embedding(input_dim=4096, output_dim=1))
    #model.add(tf.keras.layers.LSTM(128))
    #model.add(tf.keras.layers.LSTM(128))
    model.add(tf.keras.layers.Conv1D(32, 9, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(100, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid))

    # model = tf.keras.Sequential()
    # model.add(tf.keras.layers.Embedding(input_dim=1000, output_dim=64))
    # model.add(tf.keras.layers.LSTM(128))
    # model.add(tf.keras.layers.Dense(10))
    
    loss = tf.keras.losses.BinaryCrossentropy()

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                  loss = loss,
                  metric=['accuracy'])
    
    model.fit(data_training, labels_train)

    model.evaluate(data_test, labels_test)



    model.summary()
    print(data_training.shape)