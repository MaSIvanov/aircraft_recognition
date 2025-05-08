from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from tensorflow import keras


def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(450))
    model.add(Activation('relu'))
    model.add(Dropout(.65))
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['acc'])
    return model


def train_model(model, x_train, y_train, epochs=30, batch_size=64):
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        validation_split=0.3,
        batch_size=batch_size
    )
    return model, history


def save_model(model, filepath):
    model.save(filepath)


def load_model(filepath):
    from tensorflow.keras.models import load_model
    return load_model(filepath)
