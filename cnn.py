# Time-CNN
import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time


def plot_epochs_metric(hist, file_name, metric='acc'):
    plt.figure()
    plt.plot(hist.history[metric])
    plt.title('model '+metric)
    plt.ylabel(metric,fontsize='large')
    plt.xlabel('epoch',fontsize='large')
    plt.savefig(file_name,bbox_inches='tight')
    plt.close()

class Classifier_CNN:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False):
        self.output_directory = output_directory
        self.model = self.build_model(input_shape, nb_classes)
        if (verbose == True):
            self.model.summary()
        self.verbose = verbose
        self.model.save_weights(self.output_directory + 'model_init.hdf5')

    def build_model(self, input_shape, nb_classes):
        padding = 'valid'
        input_layer = keras.layers.Input(input_shape)

        if input_shape[0] < 60: # for italypowerondemand dataset
            padding = 'same'

        conv1 = keras.layers.Conv1D(filters=6,kernel_size=7,padding=padding,activation='sigmoid')(input_layer)
        conv1 = keras.layers.AveragePooling1D(pool_size=3)(conv1)

        conv2 = keras.layers.Conv1D(filters=12,kernel_size=7,padding=padding,activation='sigmoid')(conv1)
        conv2 = keras.layers.AveragePooling1D(pool_size=3)(conv2)

        flatten_layer = keras.layers.Flatten()(conv2)

        output_layer = keras.layers.Dense(units=nb_classes,activation='sigmoid')(flatten_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

        file_path = self.output_directory + 'best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                           save_best_only=True)

        self.callbacks = [model_checkpoint]

        return model

    def fit(self, x_train, y_train, x_val, y_val, x_input, batch_size=10, nEpochs=3500):
        # x_val and y_val are only used to monitor the test loss and NOT for training
        mini_batch_size = batch_size
        nb_epochs = nEpochs

        start_time = time.time()

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                              verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)

        duration = time.time() - start_time

        model = keras.models.load_model(self.output_directory + 'best_model.hdf5')

        y_pred = model.predict(x_input)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        hist_df = pd.DataFrame(hist.history)
        index_best_model = hist_df['loss'].idxmin()
        row_best_model = hist_df.loc[index_best_model]
        val_acc = row_best_model['val_acc']
        acc = row_best_model['acc']

        result = 0
        if y_pred == 1:
            result = val_acc
        else:
            result = 1 - val_acc

        plot_epochs_metric(hist, self.output_directory + 'accuracy.png')
        keras.backend.clear_session()

        return result, acc


