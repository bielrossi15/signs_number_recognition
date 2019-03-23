import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
import h5py

PATH = 'datasets/'

class Callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')> 0.99):
            print('\n\n\nAccuracy power is over 99!!!!\n\n')
            self.model.stop_training = True

def load_and_preprocess_dataset():
    """
    Function that loads and preprocess your dataset
    """

    # loading your train dataset and creating your feature and label array
    train_dataset = h5py.File(PATH + 'train_signs.h5', 'r')
    train_x = np.array(train_dataset['train_set_x'][:])
    train_y = np.array(train_dataset['train_set_y'][:])

    # loading your test dataset and creating your feature and label test array
    test_dataset = h5py.File(PATH + 'test_signs.h5', 'r')
    test_x = np.array(test_dataset['test_set_x'][:])
    test_y = np.array(test_dataset['test_set_y'][:])

    # normalizing your features
    train_x = train_x / 255
    test_x = test_x / 255

    return train_x, train_y, test_x, test_y

def model(epochs):
    train_x, train_y, test_x, test_y = load_and_preprocess_dataset()
    callback = Callback()
    model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=500, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=100, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=train_y.shape[0], activation=tf.nn.softmax)
            ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    m = model.fit(train_x, train_y, epochs=epochs, callbacks=[callback])
    model.evaluate(test_x, test_y)

    fig = plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(m.history['loss'])
    plt.xlabel('loss')
    plt.ylabel('epoch')
    plt.subplot(2, 1, 2)
    plt.plot(m.history['acc'])
    plt.xlabel('accuracy')
    plt.ylabel('epoch')    
    plt.show()

if __name__ == "__main__":
    epochs = int(input())
    model(epochs)