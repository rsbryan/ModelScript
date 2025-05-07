model RNNTextClassifier {
  // RNN model for text classification
  dataset {
    source: "custom"
    max_sequence_length: 100
    vocab_size: 10000
    num_classes: 2
  }

  architecture {
    layer Embedding {
      input_dim: 10000
      output_dim: 128
      input_length: 100
    }
    
    layer LSTM {
      units: 64
      return_sequences: false
    }
    
    layer Dense {
      units: 32
      activation: "relu"
    }
    
    layer Dropout {
      rate: 0.3
    }
    
    layer Dense {
      units: 2
      activation: "softmax"
    }
  }

  training {
    batch_size: 32
    epochs: 5
    optimizer: "adam"
    loss: "categorical_crossentropy"
    metrics: ["accuracy"]
  }

  evaluation {
    metrics: ["accuracy", "precision", "recall"]
  }
} 