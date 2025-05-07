import numpy as np
from tensorflow.keras import layers, models

# Dummy model for digit prediction
def build_model():
    model = models.Sequential([
        layers.Flatten(input_shape=[28, 28, 1]),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_model()

# Dummy training data
x_train = np.random.random((1000, 28, 28, 1))
y_train = np.random.randint(0, 10, size=(1000,))
model.fit(x_train, y_train, epochs=10, verbose=1)

# Dummy test sample (random image)
sample = np.random.random((1, 28, 28, 1))
prediction = model.predict(sample)
predicted_digit = np.argmax(prediction)
print("Number prediction probabilities:", prediction)
print("Predicted digit:", predicted_digit) 