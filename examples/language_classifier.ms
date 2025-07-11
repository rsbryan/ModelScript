model LanguageClassifier {
  dataset {
    source: "language"
    max_sequence_length: 100
    vocab_size: 5000
    num_classes: 3
  }

  architecture {
    layer Embedding {
      input_dim: 5000
      output_dim: 64
      input_length: 100
      mask_zero: true
    }
    layer GlobalAveragePooling1D {
    }
    layer Dropout {
      rate: 0.3
    }
    layer Dense {
      units: 32
      activation: "relu"
    }
    layer Dense {
      units: 3
      activation: "softmax"
    }
  }

  training {
    batch_size: 32
    epochs: 30
    optimizer: "adam"
    loss: "sparse_categorical_crossentropy"
    metrics: ["accuracy"]
    validation_split: 0.2
  }

  evaluation {
    metrics: ["accuracy"]
  }
}