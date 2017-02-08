import numpy
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from ketas.layers.embeddings import Embedding
from keras.preprocessing import sequence

#fix random number for repeatability
numpy.random.seed(17)

#def _load_data(data):
"""
data should be pd.DataFrame()
"""

input_dim = 3
max_shot_count = 500
max_shot_length = 2000
output_class_count = 4
angle_vector_length = x

(x_train_upper, x_train_lower, y_train), (x_test_upper, x_test_lower, y_test) = shotdata.load_shot(max_shot_length)

#Upper hand LSTM input
encoder_a = Sequential()
encoder_a.add(LSTM(output_dim=2*input_dim, input_shape=(max_shot_length, input_dim), activation='sigmoid', inner_activation='hard_sigmoid'))
#Lower hand LSTM input
encoder_b = Sequential()
encoder_b.add(LSTM(output_dim=2*input_dim, input_shape=(max_shot_length, input_dim), activation='sigmoid', inner_activation='hard_sigmoid'))

#Merge both LSTM units with concatenation
decoder = Sequential()
decoder.add(Merge([encoder_a, encoder_b], mode='concat'))
#Use CNNs to reduce to desired output classes
decoder.add(Dense(output_dim=2*input_dim, activation='relu'))
decoder.add(Dense(output_class_count, activation='softmax'))
decoder.compile(loss='categorical_crossentry', optimizer='rmsprop', metrics=['accuracy'])

decoder.fit([x_train_upper, x_train_lower], y_train,
            batch_size=4*input_dim, nb_epoch=20,
            validation_data=([x_test_upper, x_test_lower], y_test))

print(decoder.summary())
#model.add(Embedding(max_shot_count, angle_vector_length, input_length=max_shot_length))
#model.add(LSTM(output_dim=3, activation='sigmoid', inner_activation='hard_sigmoid'))
#model.add(Dropout(0.5))
#model.add(Dense(1))
#model.add(Activation('sigmoid'))
#model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#model.fit(x_train, y_train, validation_data=(x_test, y_test), nb_epoch=20, batch_size=4*input_dim)

scores = decoder.evaluate(x_test, y_test, verbose=0, batch_size=16)
print("Accuracy: %2f%%" % (scores[1]*100))
