#!/usr/bin/env python
"""
ModelScript - A simple language for defining machine learning models
"""
import argparse
import os
import sys
import re
import json
import importlib

def parse_model_script(file_path):
    """
    Parse a ModelScript file and return a structured representation
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract the model definition
    model_match = re.search(r'model\s+(\w+)\s*{(.*)}', content, re.DOTALL)
    if not model_match:
        raise ValueError("Invalid ModelScript: No model definition found")
    
    model_name = model_match.group(1)
    model_body = model_match.group(2)
    
    # We'll manually parse the top-level sections first
    sections = {}
    
    # Extract dataset section
    dataset_match = re.search(r'dataset\s*{(.*?)}', model_body, re.DOTALL)
    if dataset_match:
        sections['dataset'] = parse_key_value_pairs(dataset_match.group(1))
    
    # Extract architecture section with its nested layer definitions
    architecture_match = re.search(r'architecture\s*{(.*?training)', model_body, re.DOTALL)
    if architecture_match:
        # Remove the 'training' part we captured at the end
        architecture_content = architecture_match.group(1).rsplit('}', 1)[0] + '}'
        architecture = {'layers': []}
        
        # Extract all layer definitions
        layer_pattern = re.compile(r'layer\s+(\w+)\s*{(.*?)}', re.DOTALL)
        layer_matches = list(layer_pattern.finditer(architecture_content))
        
        for layer_match in layer_matches:
            layer_type = layer_match.group(1)
            layer_params = parse_key_value_pairs(layer_match.group(2))
            architecture['layers'].append({layer_type: layer_params})
        
        sections['architecture'] = architecture
    
    # Extract training section
    training_match = re.search(r'training\s*{(.*?)}', model_body, re.DOTALL)
    if training_match:
        sections['training'] = parse_key_value_pairs(training_match.group(1))
    
    # Extract evaluation section
    evaluation_match = re.search(r'evaluation\s*{(.*?)}', model_body, re.DOTALL)
    if evaluation_match:
        sections['evaluation'] = parse_key_value_pairs(evaluation_match.group(1))
    
    return {
        "name": model_name,
        "sections": sections
    }

def parse_key_value_pairs(content):
    """
    Parse key-value pairs from a section
    """
    properties = {}
    
    # Handle simple key-value pairs
    kv_pattern = re.compile(r'(\w+):\s*([^;\n]+)', re.DOTALL)
    for kv_match in kv_pattern.finditer(content):
        key = kv_match.group(1)
        value = kv_match.group(2).strip()
        
        # Try to parse array values
        if value.startswith('[') and value.endswith(']'):
            try:
                value = json.loads(value)
            except:
                pass
        # Try to parse numeric values
        elif value.isdigit():
            value = int(value)
        elif re.match(r'^-?\d+(\.\d+)?$', value):
            value = float(value)
        # Handle quoted strings
        elif value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        
        properties[key] = value
    
    return properties

# Keep the original parse_section for backward compatibility
def parse_section(section_content):
    return parse_key_value_pairs(section_content)

def generate_python_code(model_definition):
    """
    Generate executable Python code from the model definition
    """
    model_name = model_definition["name"]
    sections = model_definition["sections"]
    
    # Generate code for a Keras model
    code = [
        "import tensorflow as tf",
        "from tensorflow.keras import layers, models",
        "import numpy as np",
        ""
    ]
    
    # Function to build the model
    code.append(f"def build_{model_name.lower()}():")
    
    # Dataset handling
    if "dataset" in sections:
        dataset = sections["dataset"]
        code.append(f"    # Load dataset: {dataset.get('source', 'custom')}")
        
        if dataset.get("source") == "mnist":
            code.append("    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()")
            code.append("    # Normalize pixel values")
            code.append("    x_train, x_test = x_train / 255.0, x_test / 255.0")
            code.append("")
        elif dataset.get("source") == "cifar10":
            code.append("    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()")
            code.append("    # Normalize pixel values")  
            code.append("    x_train, x_test = x_train / 255.0, x_test / 255.0")
            code.append("")
        elif dataset.get("source") == "language":
            code[0:0] = [
                "from sklearn.preprocessing import LabelEncoder",
                "import pandas as pd",
                "from tensorflow.keras.preprocessing.text import Tokenizer",
                "from tensorflow.keras.preprocessing.sequence import pad_sequences"
            ]
            code.append("    df = pd.read_csv('data/language_data.csv')")
            code.append("    texts = df['text'].astype(str).tolist()")
            code.append("    le = LabelEncoder()")
            code.append("    labels = le.fit_transform(df['label'])")
            code.append("    max_words = 10000")
            code.append("    max_len = 100")
            code.append("    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')")
            code.append("    tokenizer.fit_on_texts(texts)")
            code.append("    sequences = tokenizer.texts_to_sequences(texts)")
            code.append("    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')")
            code.append("    labels = np.array(labels)")
            code.append("    split = int(0.8 * len(padded))")
            code.append("    x_train, y_train = padded[:split], labels[:split]")
            code.append("    x_test, y_test = padded[split:], labels[split:]")
            code.append("")
        elif dataset.get("source") == "sentiment":
            code[0:0] = [
                "import pandas as pd",
                "from tensorflow.keras.preprocessing.text import Tokenizer",
                "from tensorflow.keras.preprocessing.sequence import pad_sequences"
            ]
            code.append("    df = pd.read_csv('data/sentiment_data.csv')")
            code.append("    texts = df['text'].astype(str).tolist()")
            code.append("    labels = df['sentiment'].astype(int).tolist()")
            code.append("    max_words = 10000")
            code.append("    max_len = 100")
            code.append("    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')")
            code.append("    tokenizer.fit_on_texts(texts)")
            code.append("    sequences = tokenizer.texts_to_sequences(texts)")
            code.append("    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')")
            code.append("    labels = np.array(labels)")
            code.append("    split = int(0.8 * len(padded))")
            code.append("    x_train, y_train = padded[:split], labels[:split]")
            code.append("    x_test, y_test = padded[split:], labels[split:]")
            code.append("")
        else:
            code.append("    # Add your text data loading or preprocessing here")
            code.append("")
    
    # Architecture
    code.append("    # Build model architecture")
    code.append("    model = models.Sequential()")
    
    if "architecture" in sections and "layers" in sections["architecture"]:
        for layer in sections["architecture"]["layers"]:
            layer_type = list(layer.keys())[0]
            layer_params = layer[layer_type]
            
            if layer_type == "Bidirectional":
                # Extract inner layer type and params
                inner_layer_type = layer_params.get("layer", "LSTM")
                inner_params = {k: v for k, v in layer_params.items() if k != "layer"}
                inner_params_str = ", ".join([f"{k}={repr(v)}" for k, v in inner_params.items()])
                code.append(f"    model.add(layers.Bidirectional(layers.{inner_layer_type}({inner_params_str})))")
            else:
                params_str = ", ".join([f"{k}={repr(v)}" for k, v in layer_params.items()])
                code.append(f"    model.add(layers.{layer_type}({params_str}))")
    
    # Training configuration
    if "training" in sections:
        training = sections["training"]
        code.append("")
        code.append("    # Compile model")
        optimizer = training.get("optimizer", "adam")
        # Force sparse_categorical_crossentropy for MNIST
        if dataset.get("source") == "mnist":
            loss = "sparse_categorical_crossentropy"
        else:
            loss = training.get("loss", "sparse_categorical_crossentropy")
        metrics = training.get("metrics", ["accuracy"])
        epochs = training.get("epochs", 5)
        metrics_str = "[" + ", ".join([f"'{m}'" for m in metrics]) + "]"
        code.append(f"    model.compile(optimizer='{optimizer}', loss='{loss}', metrics={metrics_str})")
        code.append("")
        code.append(f"    model.fit(x_train, y_train, epochs={epochs}, validation_data=(x_test, y_test))")
        code.append("")
        code.append("    # Example: Make a prediction")
        code.append("    sample = x_test[0:1]")
        code.append("    prediction = model.predict(sample)")
        code.append("    guess = np.argmax(prediction, axis=1)[0]")
        if dataset.get("source") == "sentiment":
            code.append('    class_names = ["Negative", "Positive"]')
            code.append('    print("Predicted sentiment:", class_names[guess])')
        elif dataset.get("source") == "language":
            code.append('    class_names = ["English", "French", "Spanish"]  # Update as needed')
            code.append('    print("Predicted language:", class_names[guess])')
        elif dataset.get("source") == "mnist":
            code.append('    print("Predicted digit:", guess)')
        code.append('    print("Prediction:", prediction)')
    
    # Return the model
    code.append("")
    code.append("    return model")
    
    # Main execution
    code.extend([
        "",
        "if __name__ == '__main__':",
        f"    model = build_{model_name.lower()}()",
        "    model.summary()",
        "",
        "    # Add training code here if needed",
        "    # model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))"
    ])
    
    return "\n".join(code)

def main():
    parser = argparse.ArgumentParser(description="ModelScript - ML model definition language")
    parser.add_argument("file", help="ModelScript file to process")
    parser.add_argument("--output", "-o", help="Output Python file")
    parser.add_argument("--run", "-r", action="store_true", help="Run the generated code")
    
    args = parser.parse_args()
    
    try:
        model_def = parse_model_script(args.file)
        python_code = generate_python_code(model_def)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(python_code)
            print(f"Generated Python code saved to {args.output}")
        else:
            print(python_code)
        
        if args.run:
            # Create a temporary file to execute
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as temp_file:
                temp_path = temp_file.name
                
                # Modify the code to use dummy data to avoid download issues
                modified_code = python_code.replace(
                    "# Load dataset: mnist\n    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()",
                    """# Dummy data for testing\n    print("Creating dummy data instead of downloading dataset")
    x_train = np.random.random((1000, 28, 28))
    y_train = np.random.randint(0, 10, size=(1000,))
    x_test = np.random.random((200, 28, 28))
    y_test = np.random.randint(0, 10, size=(200,))"""
                )
                
                modified_code = modified_code.replace(
                    "# Load dataset: cifar10\n    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()",
                    """# Dummy data for testing\n    print("Creating dummy data instead of downloading dataset")
    x_train = np.random.random((1000, 32, 32, 3))
    y_train = np.random.randint(0, 10, size=(1000, 1))
    x_test = np.random.random((200, 32, 32, 3))
    y_test = np.random.randint(0, 10, size=(200, 1))"""
                )
                
                # Write the modified code to the temp file
                temp_file.write(modified_code)
            
            # Execute the temporary file
            print(f"\nExecuting model...")
            import subprocess
            result = subprocess.run([sys.executable, temp_path], check=True)
            
            # Clean up
            import os
            os.unlink(temp_path)
            
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())