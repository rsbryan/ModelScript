model SentimentClassifier {
  dataset {
    source: "sentiment"
    max_sequence_length: 100
    vocab_size: 10000
    num_classes: 2
  }

  architecture {
    layer Embedding {
      input_dim: 10000
      output_dim: 64
      input_length: 100
      mask_zero: true
    }
    layer Conv1D {
      filters: 64
      kernel_size: 3
      padding: "same"
      activation: "relu"
    }
    layer GlobalAveragePooling1D {
    }
    layer Dropout {
      rate: 0.3
    }
    layer Dense {
      units: 16
      activation: "relu"
    }
    layer Dense {
      units: 1
      activation: "sigmoid"
    }
  }

  training {
    batch_size: 32
    epochs: 30
    optimizer: "adam"
    loss: "binary_crossentropy"
    metrics: ["accuracy"]
    validation_split: 0.2
  }

  evaluation {
    metrics: ["accuracy"]
  }
}