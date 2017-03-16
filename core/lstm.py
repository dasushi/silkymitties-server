import numpy as np
from keras.models import Sequential, Model, model_from_json
from keras.layers import Input
from keras.layers.core import Dense, Activation, Dropout, Merge, RepeatVector
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils.visualize_util import plot
import keras.backend as K
import json
import random
#from IPython.display import SVG
#from keras.utils.visualize_util import plot
from pymongo import MongoClient
from math import ceil

#fix random number for repeatability
#np.random.seed(17)

input_dim = 3 #input dimensions of fused sensor data
nb_timesteps = 2000 #maximum amount of samples/timesteps per shot
nb_output_class = 3     #slap, snap, wrist, none
nb_batch_size = 36 #number of samples per batch
train_test_ratio = 0.6
nb_input_multi = 6
model_filename = 'LSTM_silkymitties.json'
weights_filename = 'LSTM_silkymitties_weights.h5'



def load_training_data():
    client = MongoClient('localhost', 27017)
    db = client['restdb']
    handedness = []
    speed = []
    accuracy = []
    shotTypes = []
    y = []
    labelled_values = db['testlblfusedshots'].find()
    nb_input_samples = labelled_values.count()
    x_upper = np.empty(shape=[nb_input_samples,nb_timesteps,input_dim])
    x_lower = np.empty(shape=[nb_input_samples,nb_timesteps,input_dim])
    handedness = np.zeros(shape=[nb_input_samples,2], dtype='float')
    #speed = np.empty(shape=[nb_input_samples])
    #accuracy = np.empty(shape=[nb_input_samples])
    shotTypes = np.zeros(shape=[nb_input_samples,nb_output_class], dtype='float')
    #slapshotTypes = np.empty(shape=[nb_input_samples])
    #y = np.empty(shape=[nb_input_samples,nb_output_class])
    index = 0
    for target in labelled_values:
        upperTheta = np.vstack((target['upperTheta']['uthetaX'], target['upperTheta']['uthetaY'], target['upperTheta']['uthetaZ']))
        lowerTheta = np.vstack((target['lowerTheta']['lthetaX'], target['lowerTheta']['lthetaY'], target['lowerTheta']['lthetaZ']))
        normalizedUpperTheta = upperTheta / 180.0
        normalizedLowerTheta = lowerTheta / 180.0
        x_upper[index] = sequence.pad_sequences(normalizedUpperTheta, maxlen=nb_timesteps, dtype='float', padding='post', truncating='post', value=0.).T
        x_lower[index] = sequence.pad_sequences(normalizedLowerTheta, maxlen=nb_timesteps, dtype='float', padding='post', truncating='post', value=0.).T
        shotTypes[index,shotTypeToInt(target['shotType'])] = 1.0
        #slapshotTypes[index] = isSlapShot(target['shotType'])
        handedness[index,isLeftHanded(target['handedness'])] = 1.0
        #speed = nb.append(speed, target['speed']], axis=0)
        #accuracy = nb.append(accuracy, ['accuracy'], axis=0)
        index+=1

    #for size in range(20, nb_input_samples, 20):
    #    trainIndex = round(size * train_test_ratio)
    #    nb_epochs = ceil(size / nb_batch_size)
    #    trainShotTypeCompileFit(nb_epochs, handedness[0:trainIndex], handedness[trainIndex:size], \
    #            x_upper[0:trainIndex], x_lower[0:trainIndex], shotTypes[0:trainIndex,:], \
    #            x_upper[trainIndex:size], x_lower[trainIndex:size], shotTypes[trainIndex:size,:])

    #Shuffle the samples in unison to decrease data clustering
    s_handedness, s_x_upper, s_x_lower, s_shotTypes = unison_shuffle(handedness,
        x_upper, x_lower, shotTypes)

    trainIndex = round(nb_input_samples * train_test_ratio)
    nb_epochs = ceil(nb_input_samples / nb_batch_size)
    #trainSlapShotCompileFit(nb_epochs, handedness[0:trainIndex], handedness[trainIndex:], \
    #    x_upper[0:trainIndex], x_lower[0:trainIndex], slapshotTypes[0:trainIndex], \
    #    x_upper[trainIndex:], x_lower[trainIndex:], slapshotTypes[trainIndex:])
    trainShotTypeCompileFit(nb_epochs, s_handedness[0:trainIndex], s_handedness[trainIndex:], \
        s_x_upper[0:trainIndex], s_x_lower[0:trainIndex], s_shotTypes[0:trainIndex], \
        s_x_upper[trainIndex:], s_x_lower[trainIndex:], s_shotTypes[trainIndex:])

def unison_shuffle(a, b, c, d):
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p], d[p]

def trainSlapShotCompileFit(epoch_count, train_handedness, test_handedness,
    x_train_upper, x_train_lower, y_train, x_test_upper, x_test_lower, y_test):
    #Upper hand LSTM input
    encoder_a = Sequential()
    encoder_a.add(LSTM(output_dim=nb_input_multi*input_dim,
        batch_input_shape=(nb_batch_size, nb_timesteps, input_dim),
        activation='sigmoid', inner_activation='hard_sigmoid'))
    #Lower hand LSTM input
    encoder_b = Sequential()
    encoder_b.add(LSTM(output_dim=nb_input_multi*input_dim,
        batch_input_shape=(nb_batch_size, nb_timesteps, input_dim),
        activation='sigmoid', inner_activation='hard_sigmoid'))
    encoded_handedness = Sequential()
    encoded_handedness.add(keras.layers.core.RepeatVector(nb_timesteps))
    #Merge both LSTM units with concatenation
    merged = Merge([encoded_handedness, encoder_a, encoder_b], mode='concat')
    decoder = Sequential()
    decoder.add(merged)
    decoder.add(Dropout(0.1))
    decoder.add(Dense(input_dim*nb_input_multi*2 + 2, activation='relu'))
    decoder.add(Dropout(0.2))
    decoder.add(Dense(input_dim*2, activation='relu'))
    decoder.add(Dropout(0.3))
    #sigmoid activation instead of softmax to avoid normalizing to 1.0
    #1 output signal for the binary classification likelihood
    decoder.add(Dense(1, activation='sigmoid'))

    decoder.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    decoder.fit([train_handedness, x_train_upper, x_train_lower], y_train,
    batch_size=nb_batch_size, nb_epoch=epoch_count,
    validation_data=([test_handedness, x_test_upper, x_test_lower], y_test))

    printSummary(decoder,test_handedness,x_test_upper, x_test_lower, y_test)
    return decoder

def trainShotTypeCompileFit(epoch_count, train_handedness, test_handedness,
    x_train_upper, x_train_lower, y_train, x_test_upper, x_test_lower, y_test):
    #Upper hand LSTM input
    encoder_a = Sequential()
    encoder_a.add(LSTM(4*input_dim,
        batch_input_shape=(nb_batch_size, nb_timesteps, input_dim),
        activation='sigmoid', inner_activation='hard_sigmoid',
        return_sequences=True))
    encoder_a.add(Dropout(0.2))
    encoder_a.add(LSTM(4*input_dim, return_sequences=True,
        activation='sigmoid', inner_activation='hard_sigmoid'))
    encoder_a.add(Dropout(0.25))
    encoder_a.add(LSTM(8*input_dim, batch_input_shape=(nb_batch_size, nb_timesteps, input_dim),
        activation='sigmoid', inner_activation='hard_sigmoid'))

    #Lower hand LSTM input
    encoder_b = Sequential()
    encoder_b.add(LSTM(4*input_dim,
        batch_input_shape=(nb_batch_size, nb_timesteps, input_dim),
        activation='sigmoid', inner_activation='hard_sigmoid',
        return_sequences=True))
    encoder_b.add(Dropout(0.2))
    encoder_b.add(LSTM(4*input_dim, return_sequences=True,
        activation='sigmoid', inner_activation='hard_sigmoid'))
    encoder_b.add(Dropout(0.25))
    encoder_b.add(LSTM(8*input_dim, batch_input_shape=(nb_batch_size, nb_timesteps, input_dim),
        activation='sigmoid', inner_activation='hard_sigmoid'))

    encoded_handedness = Sequential()
    encoded_handedness.add(Dense(2, batch_input_shape=(nb_batch_size, 2)))

    #Merge both LSTM units with concatenation
    merged = Merge([encoded_handedness, encoder_a, encoder_b], mode='concat')
    decoder = Sequential()
    decoder.add(merged)
    #decoder.add(Dropout(0.25))
    #Use CNNs to expand then shrink to desired output signal
    decoder.add(Dropout(0.25))
    decoder.add(Dense(input_dim*8, activation='sigmoid'))
    decoder.add(Dropout(0.25))
    decoder.add(Dense(output_dim=(2*nb_output_class), activation='sigmoid'))
    decoder.add(Dropout(0.3))
    decoder.add(Dense(nb_output_class, activation='softmax'))

    decoder.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    decoder.fit([train_handedness, x_train_upper, x_train_lower], y_train,
    batch_size=nb_batch_size, nb_epoch=epoch_count,
    validation_data=([test_handedness, x_test_upper, x_test_lower], y_test))

    printSummary(decoder, test_handedness, x_test_upper, x_test_lower, y_test)
    saveCompiledShotTypeModel(decoder)
    return decoder

def saveCompiledShotTypeModel(decoder):
    saved_model = decoder.to_json()
    with open('LSTM_silkymitties_ShotType.json', 'w') as outfile:
        json.dump(saved_model, outfile)
    decoder.save_weights('LSTM_silkymitties_ShotType_weights.h5')

def loadCompiledShotTypeModel():
    with open('LSTM_silkymitties_ShotType.json', 'r') as infile:
        architecture = json.load(infile)
    model = model_from_json(architecture)
    model.load_weights('LSTM_silkymitties_ShotType_weights.h5')
    return model

def predictShotTypeResult(fusedShotID):
    client = MongoClient('localhost', 27017)
    db = client['restdb']
    shot = db['fusedshots'].find_one(fusedShotID)
    x_upper = np.empty(shape=[1,nb_timesteps,input_dim])
    x_lower = np.empty(shape=[1,nb_timesteps,input_dim])
    raw_x_upper = np.vstack((shot['upperTheta']['uthetaX'],
            shot['upperTheta']['uthetaY'], shot['upperTheta']['uthetaZ']))
    raw_x_lower = np.vstack((shot['lowerTheta']['lthetaX'],
            shot['lowerTheta']['lthetaY'], shot['lowerTheta']['lthetaZ']))
    normalizedUpperTheta = raw_x_upper / 180.0
    normalizedLowerTheta = raw_x_lower / 180.0
    x_upper[0] = sequence.pad_sequences(normalizedUpperTheta, maxlen=nb_timesteps,
                dtype='float', padding='post', truncating='post', value=0.).T
    x_lower[0] = sequence.pad_sequences(normalizedLowerTheta, maxlen=nb_timesteps,
                dtype='float', padding='post', truncating='post', value=0.).T
    handedness = np.zeros(shape=[1,2])
    handedness[0,isLeftHanded(shot['handedness'])] = 1.0

    print("Loading Model")
    model = loadCompiledShotTypeModel()
    print("Loaded Model Succesfully")
    result = model.predict([handedness, x_upper, x_lower], batch_size=1)
    print("Result: " + str(result))
    resultIndex = result.argmax()
    print(resultIndex)
    shotTypeResult = shotTypeToString(resultIndex)
    print(shotTypeResult)
    return shotTypeResult

def trainShotSpeedCompileFit(speed, test_handedness, x_train_upper, x_train_lower, y_train, x_test_upper, x_test_lower, y_test):
    #Upper hand LSTM input
    encoder_a = Sequential()
    encoder_a.add(LSTM(output_dim=2*input_dim,
                    input_shape=(nb_timesteps, input_dim),
                    activation='sigmoid', inner_activation='hard_sigmoid'))
    #Lower hand LSTM input
    encoder_b = Sequential()
    encoder_b.add(LSTM(output_dim=2*input_dim,
                    input_shape=(nb_timesteps, input_dim),
                    activation='sigmoid', inner_activation='hard_sigmoid'))

    #Merge both LSTM units with concatenation
    merged = Merge([speed, encoder_a, encoder_b], mode='concat')
    decoder = Sequential()
    decoder.add(merged)
    #Use CNNs to reduce to desired intermediate shape
    decoder.add(Dense(output_dim=2*input_dim, activation='relu'))
    #sigmoid activation instead of softmax to avoid normalizing to 1.0
    #1 output signal for the binary classification likelihood
    decoder.add(Dense(1, activation='sigmoid'))

    decoder.compile(loss='mae', optimizer='rmsprop', metrics=['accuracy'])
    decoder.fit([x_train_upper, x_train_lower], y_train,
                batch_size=4*input_dim, nb_epoch=nb_epoch,
                validation_data=([x_test_upper, x_test_lower], y_test))

    printSummary(decoder,x_test_upper, x_test_lower, y_test)
    return decoder

def saveCompiledShotSpeedModel(decoder):
    saved_model = decoder.to_json()
    with open('LSTM_silkymitties_ShotSpeed.json', 'w') as outfile:
        json.dump(saved_model, outfile)
    decoder.save_weights('LSTM_silkymitties_ShotSpeed_weights.h5')

def loadCompiledShotSpeedModel():
    with open('LSTM_silkymitties_ShotSpeed.json', 'r') as infile:
        architecture = json.load(infile)
    model = model_from_json(architecture)
    model.load_weights('LSTM_silkymitties_ShotSpeed_weights.h5')
    return model

def trainShotAccuracyCompileFit(accuracy, test_handedness, x_train_upper,
                x_train_lower, y_train, x_test_upper, x_test_lower, y_test):
    #Upper hand LSTM input
    encoder_a = Sequential()
    encoder_a.add(LSTM(output_dim=2*input_dim,
                input_shape=(nb_timesteps, input_dim),
                activation='sigmoid', inner_activation='hard_sigmoid'))
    #Lower hand LSTM input
    encoder_b = Sequential()
    encoder_b.add(LSTM(output_dim=2*input_dim,
                input_shape=(nb_timesteps, input_dim),
                activation='sigmoid', inner_activation='hard_sigmoid'))

    #Merge both LSTM units with concatenation
    merged = Merge([accuracy, encoder_a, encoder_b], mode='concat')
    decoder = Sequential()
    decoder.add(merged)
    #Use CNNs to reduce to desired output classes
    decoder.add(Dense(output_dim=2*input_dim, activation='relu'))
    #decoder.add(Dropout(0.5))
    decoder.add(Dense(1, activation='softmax'))

    decoder.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
    decoder.fit([x_train_upper, x_train_lower], y_train,
                batch_size=4*input_dim, nb_epoch=nb_epoch,
                validation_data=([x_test_upper, x_test_lower], y_test))

    printSummary(decoder,x_test, y_test)
    return decoder

def saveCompiledShotAccuracyModel(decoder):
    saved_model = decoder.to_json()
    with open('LSTM_silkymitties_ShotAccuracy.json', 'w') as outfile:
        json.dump(saved_model, outfile)
    decoder.save_weights('LSTM_silkymitties_ShotAccuracy_weights.h5')

def loadCompiledShotAccuracyModel():
    with open('LSTM_silkymitties_ShotAccuracy.json', 'r') as infile:
        architecture = json.load(infile)
    model = model_from_json(architecture)
    model.load_weights('LSTM_silkymitties_ShotAccuracy_weights.h5')
    return model


def mean_pred(y_true, y_pred):
    return K.mean(y_pred)


def isLeftHanded(handedness):
    if str(handedness) == "L":
        return 1
    else:
        return 0

def isSlapShot(shotType):
    if str(shotType) == "slap" or str(shotType) == "Slap":
        return 1
    else:
        return 0

def shotTypeToInt(shotType):
    stringShot = str(shotType)
    #if stringShot == "notashot" or stringShot == "none" or stringShot == "NoShot":
    #    return 0
    if stringShot == "slap" or stringShot == "Slap":
        return 0
    elif stringShot == "wrist" or stringShot == "Wrist":
        return 1
    elif stringShot == "snap" or stringShot == "Snap":
        return 2

def shotTypeToString(shotType):
    #if stringShot == "notashot" or stringShot == "none" or stringShot == "NoShot":
    #    return 0
    if shotType == 0:
        return "Slap"
    elif shotType == 1:
        return "Wrist"
    elif shotType == 2:
        return "Snap"


def printSummary(decoder, test_handedness, x_test_upper, x_test_lower, y_test):
    #print(decoder.summary())
    scores = decoder.evaluate([test_handedness, x_test_upper, x_test_lower],
                y_test, verbose=0, batch_size=nb_batch_size)
    print("Accuracy: %2f%%" % (scores[1]*100))
    print(scores)
    plot(decoder, to_file='model.png', show_shapes=True)
    #plot(decoder, to_file='model.png')

if __name__=='__main__':
    #loadFromFile(True)
    #fuseLabelledShots()
    load_training_data()
