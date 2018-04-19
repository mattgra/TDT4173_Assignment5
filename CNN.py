import numpy as np
import keras
from load_data import load_images
from matplotlib import pyplot as plt
from utils import plot_random_letters

#!source activate tensorflow


def plot_training_score(history):
    
    # plot accuracy vs. epochs
    plt.ion()
    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='upper left')
    plt.show()
    
    # plot loss vs. epochs
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='upper left')
    plt.show()


def load_model():

    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    # # evaluate loaded model on test data
    loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return loaded_model


if __name__ == '__main__':

    model = keras.Sequential()
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation = 'relu', input_shape=(20,20,1)))
    model.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation = 'relu'))
    model.add(keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation = 'relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(units=27, activation = 'softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])



    train_img, test_img, train_label, test_label = load_images(flatten=False)
    train_label_one_hot = keras.utils.to_categorical(train_label, num_classes=27) # one for background
    test_label_one_hot = keras.utils.to_categorical(test_label, num_classes=27)   # one for background

    train_img = train_img.reshape(len(train_img), 20, 20, 1)/255.
    test_img = test_img.reshape(len(test_img), 20, 20, 1)/255.

    history = model.fit(x=train_img, y=train_label_one_hot, 
                        verbose=1, 
                        batch_size=32, 
                        epochs=30, 
                        validation_split=0.2)
                        #validation_data=(test_img, test_label_one_hot))
    plot_training_score(history)


    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")


    # show some wrong predictions
    if False:
        prediction = model.predict_classes(test_img).reshape((-1,))
        incorrects = np.where(prediction != test_label)
        plot_random_letters(test_img[incorrects], prediction[incorrects])


