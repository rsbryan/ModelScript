import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Dummy sentiment model
def build_model():
    model = models.Sequential([
        layers.Embedding(input_dim=1000, output_dim=16, input_length=10),
        layers.LSTM(8),
        layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_model()

# Toy training data
texts = ["I love this!", "I hate this!", "This is great!", "This is terrible!"]
labels = [1, 0, 1, 0]  # 1=positive, 0=negative
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=10)
labels = np.array(labels)
model.fit(padded, labels, epochs=5, verbose=1)

# Sample input
sample_text = "This is amazing!"
seq = tokenizer.texts_to_sequences([sample_text])
padded_sample = pad_sequences(seq, maxlen=10)
prediction = model.predict(padded_sample)
labels_str = ["negative", "positive"]
predicted_sentiment = labels_str[np.argmax(prediction)]
print("Sentiment prediction probabilities:", prediction)
print("Predicted sentiment:", predicted_sentiment) 