import numpy
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import keras.backend as K
import json
#from IPython.display import SVG
#from keras.utils.visualize_util import plot
import pymongo

#fix random number for repeatability
numpy.random.seed(17)

input_dim = 3 #input dimensions of fused sensor data
nb_timesteps = 1000 #maximum amount of samples/timesteps per shot
nb_output_class = 4     #slap, snap, wrist, none
nb_epoch = 10 #number of epochs
nb_batch_size = 10 #number of batches of epochs
train_test_ratio = 0.7
model_filename = 'LSTM_silkymitties.json'
weights_filename = 'LSTM_silkymitties_weights.h5'



def load_training_data():
    client = MongoClient('localhost', 27017)
    db = client['restdb']
    x_upper = np.empty(shape=[0,nb_timesteps,input_dim])
    x_lower = np.empty(shape=[0,nb_timesteps,input_dim])
    handedness = np.empty(shape=[0,1])
    speed = np.empty(shape=[0,1])
    accuracy = np.empty(shape=[0,1])
    y= np.empty(shape=[0,1])
    count = 0
    for target in db['labelledshots'].find():
        paddedUpper = sequence.pad_sequences(target['upperTheta'], maxlen=nb_timesteps, dtype='float', padding='post', truncating='post', value=0.)
        paddedLower = sequence.pad_sequences(target['lowerTheta'], maxlen=nb_timesteps, dtype='float', padding='post', truncating='post', value=0.)
        x_upper = np.append(x_upper, [paddedUpper], axis=0)
        x_lower = np.append(x_lower, [paddedLower], axis=0)
        y = np.append(y, [target['shotType']], axis=0)
        user = db['users'].find_one({"_id":ObjectId(target['userID'])})
        isLeftHanded = user['handedness'] == 'L' ? 1 : 0
        handedness = nb.append(shotType, [isLeftHanded], axis=0)
        speed = nb.append(speed, target['speed']], axis=0)
        accuracy = nb.append(accuracy, ['accuracy'], axis=0)
        count++
    trainIndex = round(count * train_test_ratio)
    trainShotTypeCompileFit(handedness[0:trainIndex, :], handedness[trainIndex:, :], \
            x_upper[0:trainIndex,:,:], x_lower[0:trainIndex,:,:], y[0:trainIndex,:], \
            x_upper[trainIndex:,:,:], x_lower[trainIndex:,:,:], y[trainIndex,:])



def predictShotTypeResult(fusedShotID):
    model = loadCompiledModel()


def trainShotTypeCompileFit(train_handedness, test_handedness, x_train_upper, x_train_lower, y_train, x_test_upper, x_test_lower, y_test):
    #Upper hand LSTM input
    encoder_a = Sequential()
    encoder_a.add(LSTM(output_dim=2*input_dim, input_shape=(nb_timesteps, input_dim), activation='sigmoid', inner_activation='hard_sigmoid'))
    #Lower hand LSTM input
    encoder_b = Sequential()
    encoder_b.add(LSTM(output_dim=2*input_dim, input_shape=(nb_timesteps, input_dim), activation='sigmoid', inner_activation='hard_sigmoid'))

    #Merge both LSTM units with concatenation
    merged = Merge([handedness, encoder_a, encoder_b], mode='concat')
    decoder = Sequential()
    decoder.add(merged)
    #Use CNNs to reduce to desired output classes
    decoder.add(Dense(output_dim=2*input_dim, activation='relu'))
    decoder.add(Dropout(0.5))
    decoder.add(Dense(nb_output_class, activation='softmax'))

    decoder.compile(loss='categorical_crossentry', optimizer='rmsprop', metrics=['accuracy'])
    decoder.fit([x_train_upper, x_train_lower], y_train,
                batch_size=4*input_dim, nb_epoch=nb_epoch,
                validation_data=([x_test_upper, x_test_lower], y_test))

    printSummary(decoder,x_test, y_test)
    return decoder

def saveCompiledShotTypeModel(decoder):
    saved_model = decoder.to_json()
    with open('LSTM_silkymitties_ShotType.json', 'w') as outfile:
        json.dump(saved_model, outfile)
    decoder.save_weights('LSTM_silkymitties_ShotType_weights.h5')

def loadCompiledShotTypeModel():
    with open('LSTM_silkymitties.json', 'r') as infile:
        architecture = json.load(infile)
    model = model_from_json(architecture)
    model.load_weights('LSTM_silkymitties_ShotType_weights.h5')
    return model

def predictShotSpeedResult(fusedShotID):
    model = loadCompiledShotSpeedModel()


def trainShotSpeedCompileFit(speed, test_handedness, x_train_upper, x_train_lower, y_train, x_test_upper, x_test_lower, y_test):
    #Upper hand LSTM input
    encoder_a = Sequential()
    encoder_a.add(LSTM(output_dim=2*input_dim, input_shape=(nb_timesteps, input_dim), activation='sigmoid', inner_activation='hard_sigmoid'))
    #Lower hand LSTM input
    encoder_b = Sequential()
    encoder_b.add(LSTM(output_dim=2*input_dim, input_shape=(nb_timesteps, input_dim), activation='sigmoid', inner_activation='hard_sigmoid'))

    #Merge both LSTM units with concatenation
    merged = Merge([speed, encoder_a, encoder_b], mode='concat')
    decoder = Sequential()
    decoder.add(merged)
    #Use CNNs to reduce to desired output classes
    decoder.add(Dense(output_dim=2*input_dim, activation='relu'))
    decoder.add(Dropout(0.5))
    decoder.add(Dense(1, activation='softmax'))

    decoder.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
    decoder.fit([x_train_upper, x_train_lower], y_train,
                batch_size=4*input_dim, nb_epoch=nb_epoch,
                validation_data=([x_test_upper, x_test_lower], y_test))

    printSummary(decoder,x_test, y_test)
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

def trainShotAccuracyCompileFit(accuracy, test_handedness, x_train_upper, x_train_lower, y_train, x_test_upper, x_test_lower, y_test):
    #Upper hand LSTM input
    encoder_a = Sequential()
    encoder_a.add(LSTM(output_dim=2*input_dim, input_shape=(nb_timesteps, input_dim), activation='sigmoid', inner_activation='hard_sigmoid'))
    #Lower hand LSTM input
    encoder_b = Sequential()
    encoder_b.add(LSTM(output_dim=2*input_dim, input_shape=(nb_timesteps, input_dim), activation='sigmoid', inner_activation='hard_sigmoid'))

    #Merge both LSTM units with concatenation
    merged = Merge([accuracy, encoder_a, encoder_b], mode='concat')
    decoder = Sequential()
    decoder.add(merged)
    #Use CNNs to reduce to desired output classes
    decoder.add(Dense(output_dim=2*input_dim, activation='relu'))
    decoder.add(Dropout(0.5))
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


def printSummary(decoder, x_test, y_test):
    print(decoder.summary())
    scores = decoder.evaluate(x_test, y_test, verbose=0, batch_size=nb_batch_size)
    print("Accuracy: %2f%%" % (scores[1]*100))
    #plot(decoder, to_file='model.png')


#model.add(Embedding(max_shot_count, angle_vector_length, input_length=nb_timesteps))
#model.add(LSTM(output_dim=3, activation='sigmoid', inner_activation='hard_sigmoid'))
#model.add(Dropout(0.5))
#model.add(Dense(1))
#model.add(Activation('sigmoid'))
#model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#model.fit(x_train, y_train, validation_data=(x_test, y_test), nb_epoch=20, batch_size=4*input_dim)
