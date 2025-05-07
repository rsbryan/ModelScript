import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def build_mnistclassifier():
    # Dummy data for testing - avoid downloading MNIST
    print("Creating dummy data instead of downloading MNIST")
    x_train = np.random.random((1000, 28, 28))
    y_train = np.random.randint(0, 10, size=(1000,))
    x_test = np.random.random((200, 28, 28))
    y_test = np.random.randint(0, 10, size=(200,))
    
    # Normalize pixel values
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Build model architecture
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=[28, 28, 1]))
    model.add(layers.Dense(units=128, activation='relu'))
    model.add(layers.Dense(units=10, activation='softmax'))

    # Compile model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

    # Test a prediction
    sample = x_test[0].reshape(1, 28, 28)
    prediction = model.predict(sample)
    predicted_class = np.argmax(prediction)
    print("Prediction probabilities:", prediction)
    print("Predicted class:", predicted_class)
    print("Script finished running.")

    return model

if __name__ == '__main__':
    model = build_mnistclassifier()
    model.summary()