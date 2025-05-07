# ModelScript

A simple domain-specific language (DSL) for defining machine learning models.

## Overview

ModelScript is a language designed to make defining, training, and evaluating machine learning models simpler and more accessible. It provides a clear, declarative syntax for model architecture, dataset configuration, and training parameters.

## Installation

```bash
# Clone the repository
git clone https://github.com/rsbryan/modelscript.git
cd modelscript

# Install dependencies
pip install -r requirements.txt
```

## How to Run ModelScript: Step-by-Step

### 1. Set up your environment

```bash
# (Optional but recommended) Create a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required Python packages
pip install -r modelscript/requirements.txt
```
*This sets up a clean Python environment and installs TensorFlow, numpy, and other dependencies.*

---

### 2. Generate Python code from a ModelScript file

```bash
python3 modelscript/modelscript.py modelscript/examples/mnist_classifier.ms -o mnist_model.py
```
*This reads the ModelScript file (`mnist_classifier.ms`) and creates a Python file (`mnist_model.py`) with the neural network code.*

---

### 3. Run the generated Python model

```bash
python3 mnist_model.py
```
*This runs the generated model code. You'll see the model summary and (if the script includes it) training and prediction output in your terminal.*

---

### 4. Run and execute a ModelScript file directly (no intermediate Python file needed)

```bash
python3 modelscript/modelscript.py modelscript/examples/mnist_classifier.ms --run
```
*This command parses the ModelScript file, generates the Python code, and immediately runs it—all in one step.*

---

### 5. Create and run your own model

- Write your model in a `.ms` file (e.g., `my_model.ms`)
- Generate Python code:
  ```bash
  python3 modelscript/modelscript.py my_model.ms -o my_model.py
  ```
- Run your model:
  ```bash
  python3 my_model.py
  ```

---

### 6. See the output

- All output (model summary, training progress, predictions) will appear in your terminal.
- To save the output to a file:
  ```bash
  python3 my_model.py > output.txt
  ```

---

### 7. (Optional) Edit the generated Python code

- You can open the generated `.py` file and:
  - Change the data used for training/testing
  - Add your own print statements
  - Customize how predictions are displayed

---

**Tip:**
If you want to use your own data, edit the `prepare_text_data()` function in the generated Python file to load and preprocess your dataset.

---

## Usage

### Creating a Model

Create a `.ms` file with your model definition:

```
model MyModel {
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
    loss: "categorical_crossentropy"
  }
}
```

### Processing a Model

```bash
python modelscript.py path/to/your/model.ms
```

To save the generated Python code:

```bash
python modelscript.py path/to/your/model.ms -o model.py
```

To run the model immediately:

```bash
python modelscript.py path/to/your/model.ms --run
```

## Syntax

### Model Definition

```
model ModelName {
  // Model sections go here
}
```

### Dataset Section

```
dataset {
  source: "dataset_name"  // Built-in datasets: "mnist", "cifar10"
  training_size: 60000
  test_size: 10000
  input_shape: [width, height, channels]
  num_classes: 10
}
```

### Architecture Section

```
architecture {
  layer LayerType {
    // Layer parameters
    param1: value1
    param2: value2
  }
  
  // Additional layers...
}
```

### Training Section

```
training {
  batch_size: 32
  epochs: 5
  optimizer: "adam"  // Options: "adam", "sgd", "rmsprop"
  loss: "categorical_crossentropy"
  metrics: ["accuracy"]
}
```

### Evaluation Section

```
evaluation {
  metrics: ["accuracy", "precision", "recall"]
}
```

## Examples

Check out the example models in the `examples/` directory.

## VS Code Extension

This repository includes a VS Code extension for syntax highlighting and language support. To enable it:

1. Copy the `.vscode` folder to your project
2. Install the extension through the VS Code marketplace

## License

MIT 