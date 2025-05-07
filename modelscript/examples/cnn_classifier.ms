model CNNClassifier {
  // CNN image classifier model definition
  dataset {
    source: "cifar10"
    training_size: 50000
    test_size: 10000
    input_shape: [32, 32, 3]
    num_classes: 10
  }

  architecture {
    layer Conv2D {
      filters: 32
      kernel_size: [3, 3]
      padding: "same"
      activation: "relu"
      input_shape: [32, 32, 3]
    }
    
    layer MaxPooling2D {
      pool_size: [2, 2]
    }
    
    layer Conv2D {
      filters: 64
      kernel_size: [3, 3]
      padding: "same"
      activation: "relu"
    }
    
    layer MaxPooling2D {
      pool_size: [2, 2]
    }
    
    layer Conv2D {
      filters: 128
      kernel_size: [3, 3]
      padding: "same"
      activation: "relu"
    }
    
    layer MaxPooling2D {
      pool_size: [2, 2]
    }
    
    layer Flatten {}
    
    layer Dense {
      units: 128
      activation: "relu"
    }
    
    layer Dropout {
      rate: 0.5
    }
    
    layer Dense {
      units: 10
      activation: "softmax"
    }
  }

  training {
    batch_size: 64
    epochs: 20
    optimizer: "adam"
    loss: "categorical_crossentropy"
    metrics: ["accuracy"]
  }

  evaluation {
    metrics: ["accuracy", "precision", "recall"]
  }
} 