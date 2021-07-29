from keras.layers import Dense, Conv1D, MaxPooling2D, Flatten, MaxPooling1D, Dropout, Input
from keras.models import Sequential
from keras.optimizers import Adam

def SimpleFF(input_shape: tuple) -> Sequential:
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.005), metrics=['categorical_accuracy'])
    model.summary()
    return model

def CNN1D(input_shape: tuple) -> Sequential:
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0012), metrics=['categorical_accuracy'])
    model.summary()
    return model
