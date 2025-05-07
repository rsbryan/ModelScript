import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def build_rnntextclassifier():
    # Build model architecture
    model = models.Sequential()
    model.add(layers.Embedding(input_dim=10000, output_dim=128, input_length=100))
    model.add(layers.LSTM(units=64, return_sequences=False))
    model.add(layers.Dense(units=32, activation='relu'))
    model.add(layers.Dropout(rate=0.3))
    model.add(layers.Dense(units=2, activation='softmax'))

    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Example usage with text data
def prepare_text_data():
    # This is a sample function to prepare text data
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.utils import to_categorical
    
    # Sample texts (in a real scenario, this would be loaded from files)
    texts = [
        "This movie was fantastic! I really liked it.",
        "I enjoyed this movie very much and would recommend it.",
        "This was a terrible waste of time.",
        "I hated this movie. The acting was horrible.",
        "Great film, amazing performances by all actors.",
        "I was disappointed by this film, not worth watching."
    ]
    
    # Labels: 0 for negative, 1 for positive
    labels = [1, 1, 0, 0, 1, 0]
    
    # Tokenize the text
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    
    # Pad sequences to ensure uniform length
    max_length = 100
    data = pad_sequences(sequences, maxlen=max_length)
    
    # Convert labels to categorical
    labels = to_categorical(labels)
    
    # Split into train/test (simplified for demonstration)
    train_size = int(len(data) * 0.8)
    x_train, x_test = data[:train_size], data[train_size:]
    y_train, y_test = labels[:train_size], labels[train_size:]
    
    return x_train, y_train, x_test, y_test

if __name__ == '__main__':
    model = build_rnntextclassifier()
    model.summary()
    
    # Prepare sample data
    print("\nPreparing sample text data...")
    x_train, y_train, x_test, y_test = prepare_text_data()
    
    # Train the model
    print("\nTraining the model...")
    history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), verbose=1)
    
    print("\nModel is ready for text classification!")
    
    # Make a prediction with the first test sample
    print("\nMaking a prediction with sample text...")
    prediction = model.predict(x_test[:1])
    print(f"Prediction probabilities: {prediction}")
    print(f"Predicted class: {np.argmax(prediction)}")