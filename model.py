from keras.layers import Dense, Conv1D, MaxPooling2D, Flatten, MaxPooling1D
from keras.models import Sequential

def CNN1D(input_shape: tuple) -> Sequential:
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=8, activation='relu'), input_shape=input_shape)
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(5, activation='softmax'))
    return model
