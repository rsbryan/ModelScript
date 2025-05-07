import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Dummy language model
def build_model():
    model = models.Sequential([
        layers.Embedding(input_dim=1000, output_dim=16, input_length=10),
        layers.LSTM(8),
        layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_model()

# Toy training data
texts = [
    "Hello, how are you?",      # English
    "Bonjour, comment ça va?",  # French
    "Hola, ¿cómo estás?"        # Spanish
]
labels = [0, 1, 2]  # 0=English, 1=French, 2=Spanish
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=10)
labels = np.array(labels)
model.fit(padded, labels, epochs=10, verbose=1)

# Sample input
sample_text = "Bonjour, je suis content."
seq = tokenizer.texts_to_sequences([sample_text])
padded_sample = pad_sequences(seq, maxlen=10)
prediction = model.predict(padded_sample)
labels_str = ["English", "French", "Spanish"]
predicted_language = labels_str[np.argmax(prediction)]
print("Language prediction probabilities:", prediction)
print("Predicted language:", predicted_language) 