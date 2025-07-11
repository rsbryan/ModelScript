model MNISTClassifier {
  dataset {
    source: "mnist"
    input_shape: [28, 28, 1]
    num_classes: 10
  }

  architecture {
    layer Flatten {
      input_shape: [28, 28, 1]
    }
    layer Dense {
      units: 128
      activation: "relu"
    }
    layer Dense {
      units: 10
      activation: "softmax"
    }
  }

  training {
    batch_size: 32
    epochs: 5
    optimizer: "adam"
    loss: "sparse_categorical_crossentropy"
    metrics: ["accuracy"]
  }

  evaluation {
    metrics: ["accuracy"]
  }
}