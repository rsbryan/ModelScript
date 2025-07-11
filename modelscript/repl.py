#!/usr/bin/env python3
"""
ModelScript REPL - Interactive Model Building Environment
"""
import os
import sys
import tempfile
import subprocess
from typing import Dict, List, Optional
import json

# Add the current directory to the path to import modelscript
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modelscript import parse_model_script, generate_python_code

class ModelScriptREPL:
    def __init__(self):
        self.current_model = {
            'name': 'InteractiveModel',
            'dataset': {},
            'architecture': {'layers': []},
            'training': {},
            'evaluation': {}
        }
        self.examples = self._load_examples()
        self.history = []
        self.trained_model_path = None
        # Cache for trained model and tokenizer
        self.cached_model = None
        self.cached_tokenizer = None

    def _load_examples(self):
        """Load available example models"""
        examples = {}
        examples_dir = os.path.join(os.path.dirname(__file__), '..', 'examples')
        if os.path.exists(examples_dir):
            for filename in os.listdir(examples_dir):
                if filename.endswith('.ms'):
                    name = filename[:-3]  # Remove .ms extension
                    examples[name] = os.path.join(examples_dir, filename)
        return examples

    def start(self):
        """Start the interactive REPL"""
        print("🚀 ModelScript Interactive REPL")
        print("💡 Type 'help' to see all commands")
        print("🎯 Type 'examples' to see available models")
        print("🔥 Ready to build neural networks!")
        print()
        
        while True:
            try:
                command = input("modelscript> ").strip()
                if not command:
                    continue
                    
                self.history.append(command)
                
                if command.lower() in ['quit', 'exit']:
                    print("👋 Thanks for using ModelScript!")
                    break
                elif command.lower() == 'help':
                    self.show_help()
                elif command.lower() == 'examples':
                    self.show_examples()
                elif command.startswith('load '):
                    self.cmd_load(command[5:].strip())
                elif command.lower() == 'show':
                    self.cmd_show()
                elif command.lower() == 'clear':
                    self.cmd_clear()
                elif command.lower() == 'generate':
                    self.cmd_generate()
                elif command.lower() == 'train':
                    self.cmd_train()
                elif command.lower() == 'predict':
                    self.cmd_predict()
                elif command.lower() == 'run':
                    self.cmd_run_full()
                elif command.lower() == 'status':
                    self.cmd_status()
                elif command.lower() == 'history':
                    self.show_history()
                else:
                    print(f"❌ Unknown command: {command}")
                    print("💡 Type 'help' for available commands")
                    
            except (KeyboardInterrupt, EOFError):
                print("\n👋 Thanks for using ModelScript!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    def show_help(self):
        """Display help information"""
        print("🔥 ModelScript REPL Commands:")
        print()
        print("📋 Model Management:")
        print("  examples          - Show available example models")
        print("  load <name>       - Load an example model") 
        print("  show              - Display current model definition")
        print("  clear             - Clear current model")
        print()
        print("🚀 Execution:")
        print("  predict           - Interactive prediction mode")
        print("  generate          - Show Python/TensorFlow code")
        print("  run               - Train then enter prediction mode")
        print()
        print("🔧 Utilities:")
        print("  help              - Show this help")
        print("  quit/exit         - Exit the REPL")
        print()
        print("💡 Simple workflow:")
        print("  1. 'examples' - See what models are available")
        print("  2. 'load sentiment_classifier' - Load a model")  
        print("  3. 'predict' - Test it with your own text")
        print()
        print("✨ Try this flow: examples → load language_classifier → predict")

    def show_examples(self):
        """Show available example models with descriptions"""
        print("📚 Available Example Models:")
        print()
        
        if not self.examples:
            print("❌ No examples found")
            return
        
        model_descriptions = {
            'language_classifier': {
                'desc': 'Multi-language text classifier (English, French, Spanish)',
                'type': 'Text Classification',
                'dataset': '30 multilingual samples',
                'architecture': 'Embedding → LSTM → Dense'
            },
            'sentiment_classifier': {
                'desc': 'Sentiment analysis (Positive/Negative)',
                'type': 'Text Classification', 
                'dataset': '30 sentiment samples',
                'architecture': 'Embedding → LSTM → Dense'
            },
            'mnist_classifier': {
                'desc': 'Handwritten digit recognition (0-9)',
                'type': 'Image Classification',
                'dataset': 'MNIST digits',
                'architecture': 'Flatten → Dense → Dense'
            },
            'cnn_classifier': {
                'desc': 'Convolutional neural network for CIFAR-10',
                'type': 'Image Classification',
                'dataset': 'CIFAR-10 images',
                'architecture': 'Conv2D → MaxPool → Dense'
            },
            'rnn_text_classifier': {
                'desc': 'RNN-based text classifier',
                'type': 'Text Classification',
                'dataset': 'Custom text data',
                'architecture': 'Embedding → LSTM → Dense'
            }
        }
            
        for name, path in self.examples.items():
            if name in model_descriptions:
                info = model_descriptions[name]
                print(f"  🧠 {name}")
                print(f"     {info['desc']}")
                print(f"     Type: {info['type']} | Dataset: {info['dataset']}")
                print(f"     Architecture: {info['architecture']}")
                print()
            else:
                print(f"  📄 {name}")
                print()
            
        print("💡 Usage: 'load <name>' to load a model")
        print("🔥 Recommended: 'load sentiment_classifier' or 'load language_classifier'")

    def cmd_load(self, model_name):
        """Load an example model"""
        if model_name not in self.examples:
            print(f"❌ Example '{model_name}' not found")
            print("💡 Use 'examples' to see available models")
            return
            
        try:
            model_def = parse_model_script(self.examples[model_name])
            self.current_model = {
                'name': model_def['name'],
                'dataset': model_def['sections'].get('dataset', {}),
                'architecture': model_def['sections'].get('architecture', {'layers': []}),
                'training': model_def['sections'].get('training', {}),
                'evaluation': model_def['sections'].get('evaluation', {})
            }
            
            print(f"✅ Loaded model: {model_def['name']}")
            
            # Show model capabilities
            model_type = self._detect_model_type()
            if model_type == 'language':
                print("🌍 Capabilities: Language detection (English, French, Spanish)")
                print("💡 Next: 'predict' for interactive classification")
            elif model_type == 'sentiment':
                print("😊 Capabilities: Sentiment analysis (Positive/Negative)")
                print("💡 Next: 'predict' for interactive sentiment analysis")
            elif model_type == 'mnist':
                print("🔢 Capabilities: Handwritten digit recognition (0-9)")
                print("💡 Next: 'train' to train the model, then 'predict'")
            else:
                print("🧠 Neural network model loaded")
                print("💡 Next: 'show' to see structure, 'train' to train")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")

    def cmd_show(self):
        """Display the current model definition"""
        print(f"🧠 Current Model: {self.current_model['name']}")
        print()
        
        if self.current_model['dataset']:
            print("📊 Dataset Configuration:")
            for key, value in self.current_model['dataset'].items():
                print(f"  {key}: {value}")
            print()
            
        if self.current_model['architecture']['layers']:
            print("🏗️  Neural Network Architecture:")
            for i, layer in enumerate(self.current_model['architecture']['layers']):
                layer_type = list(layer.keys())[0]
                layer_params = layer[layer_type]
                print(f"  {i+1}. {layer_type}")
                for key, value in layer_params.items():
                    print(f"     {key}: {value}")
            print()
            
        if self.current_model['training']:
            print("🎯 Training Configuration:")
            for key, value in self.current_model['training'].items():
                print(f"  {key}: {value}")
            print()
        
        # Show available actions
        model_type = self._detect_model_type()
        print("🚀 Available Actions:")
        if model_type in ['language', 'sentiment']:
            print("  'predict' - Interactive text classification")
            print("  'train' - Train neural network first (optional)")
        elif model_type == 'mnist':
            print("  'train' - Train on MNIST dataset")
            print("  'predict' - Test with sample images (after training)")
        else:
            print("  'generate' - Show Python/TensorFlow code")
            print("  'train' - Train the neural network")

    def cmd_clear(self):
        """Clear the current model"""
        self.current_model = {
            'name': 'InteractiveModel',
            'dataset': {},
            'architecture': {'layers': []},
            'training': {},
            'evaluation': {}
        }
        self.trained_model_path = None
        print("🧹 Model cleared")

    def cmd_generate(self):
        """Generate Python code from current model"""
        if self.current_model['name'] == 'InteractiveModel' and not self.current_model['architecture']['layers']:
            print("❌ No model loaded. Use 'load <model>' first.")
            print("💡 Try: 'examples' to see available models")
            return
            
        try:
            # Format model for generation
            model_for_generation = {
                'name': self.current_model['name'],
                'sections': {
                    'dataset': self.current_model['dataset'],
                    'architecture': self.current_model['architecture'],
                    'training': self.current_model['training'],
                    'evaluation': self.current_model['evaluation']
                }
            }
            
            python_code = generate_python_code(model_for_generation)
            print("🐍 Generated Python/TensorFlow Code:")
            print("=" * 60)
            print(python_code)
            print("=" * 60)
            print()
            print("💡 This code can be saved and run independently!")
            
        except Exception as e:
            print(f"❌ Error generating code: {e}")

    def cmd_status(self):
        """Show current model status and capabilities"""
        print("📊 Model Status:")
        print(f"  Current Model: {self.current_model['name']}")
        
        if self.current_model['name'] == 'InteractiveModel':
            print("  Status: ❌ No model loaded")
            print("  💡 Use 'examples' then 'load <model>' to get started")
            return
        
        model_type = self._detect_model_type()
        dataset_source = self.current_model.get('dataset', {}).get('source', 'unknown')
        
        print(f"  Dataset: {dataset_source}")
        print(f"  Type: {model_type}")
        print(f"  Layers: {len(self.current_model['architecture']['layers'])}")
        
        if self.trained_model_path:
            print("  Training: ✅ Model trained")
        else:
            print("  Training: ⚠️  Not trained yet")
        
        print()
        print("🎯 Recommended Actions:")
        if model_type == 'language':
            print("  • 'predict' - Test language detection (English/French/Spanish)")
        elif model_type == 'sentiment':
            print("  • 'predict' - Test sentiment analysis (Positive/Negative)")
        elif model_type == 'mnist':
            print("  • 'train' - Train on MNIST digits")
        else:
            print("  • 'show' - View model architecture")
            print("  • 'generate' - See Python code")

    def cmd_train(self):
        """Train the neural network"""
        if self.current_model['name'] == 'InteractiveModel':
            print("❌ No model loaded. Use 'load <model>' first.")
            return
        
        model_type = self._detect_model_type()
        
        print("🚀 Training Neural Network...")
        print(f"📊 Model: {self.current_model['name']}")
        print(f"🧠 Type: {model_type}")
        
        if model_type in ['language', 'sentiment']:
            success = self._quick_train_model()
            if success:
                print("✅ Training completed!")
                print("💡 Now try 'predict' for interactive classification")
            else:
                print("⚠️  Training had issues, but fallback mode available")
                print("💡 You can still use 'predict' with rule-based classification")
        else:
            print("🔄 Attempting to train model...")
            success = self._train_general_model()
            if success:
                print("✅ Training completed!")
            else:
                print("❌ Training failed - check dependencies")

    def cmd_predict(self):
        """🔥 CRITICAL: Interactive prediction mode"""
        if self.current_model['name'] == 'InteractiveModel':
            print("❌ No model loaded. Use 'load <model>' first.")
            print("💡 Try: 'load sentiment_classifier' or 'load language_classifier'")
            return
        
        model_type = self._detect_model_type()
        
        print(f"🔥 Starting Interactive Prediction Mode")
        print(f"📊 Model: {self.current_model['name']}")
        print()
        
        if model_type == 'language':
            self._interactive_language_prediction()
        elif model_type == 'sentiment':
            self._interactive_sentiment_prediction()
        elif model_type == 'mnist':
            self._interactive_mnist_prediction()
        else:
            print("💡 Prediction mode available for language, sentiment, and MNIST models.")
            print("💡 Current model type not supported for interactive prediction.")
            print("🔧 Try 'generate' to see the Python code for this model.")

    def cmd_run_full(self):
        """Train model then enter prediction mode"""
        if self.current_model['name'] == 'InteractiveModel':
            print("❌ No model loaded. Use 'load <model>' first.")
            return
        
        print("🚀 Full Pipeline: Train → Predict")
        print()
        
        # First train
        self.cmd_train()
        print()
        
        # Then predict
        print("🎯 Entering prediction mode...")
        print()
        self.cmd_predict()

    def _detect_model_type(self):
        """Detect the type of model based on dataset source"""
        source = self.current_model.get('dataset', {}).get('source', '')
        if source == 'language':
            return 'language'
        elif source == 'sentiment':
            return 'sentiment'
        elif source == 'custom' and self.current_model.get('dataset', {}).get('num_classes') == 2:
            return 'sentiment'
        elif source == 'mnist':
            return 'mnist'
        elif source == 'cifar10':
            return 'cnn'
        return 'general'

    def _interactive_language_prediction(self):
        """CRITICAL: Interactive language classification"""
        print("Language Classifier - Interactive Mode")
        print("I can detect if text is English, French, or Spanish!")
        print("Type some text and I'll guess the language.")
        print("Type 'back' to return to main REPL.")
        print()
        
        # Try to train model first (optional)
        print("Preparing neural network...")
        trained = self._quick_train_model()
        if trained:
            print("Neural network ready!")
        else:
            print("Neural network training failed!")
            print("You can still try manual input, but training is recommended.")
        print()
            
        while True:
            try:
                print("Enter text to classify> ", end="", flush=True)
                text = input().strip()
                if not text:
                    continue
                if text.lower() in ['back', 'exit', 'quit']:
                    print("↩️  Returning to main REPL...")
                    break
                    
                # Make prediction
                result = self._predict_language(text)
                if result:
                    language, confidence = result
                    # Fix 4: Adaptive confidence threshold based on OOV fraction
                    oov_fraction = getattr(self, '_last_oov_fraction', 0)
                    threshold = 0.4 + 0.2 * oov_fraction
                    if confidence < threshold:
                        print(f"🤔 Not sure. Best guess: {language} ({confidence:.1%} confidence) [OOV: {oov_fraction:.1%}]")
                    else:
                        print(f"Prediction: {language} ({confidence:.1%} confidence)")
                print()
                
            except (KeyboardInterrupt, EOFError):
                print("\nReturning to main REPL...")
                break

    def _interactive_sentiment_prediction(self):
        """CRITICAL: Interactive sentiment analysis"""
        print("Sentiment Analyzer - Interactive Mode")
        print("I can detect if text is positive or negative!")
        print("Type some text and I'll analyze the sentiment.")
        print("Type 'back' to return to main REPL.")
        print()
        
        # Try to train model first (optional)
        print("Preparing neural network...")
        trained = self._quick_train_model()
        if trained:
            print("Neural network ready!")
        else:
            print("Neural network training failed!")
            print("You can still try manual input, but training is recommended.")
        print()
            
        while True:
            try:
                print("Enter text to analyze> ", end="", flush=True)
                text = input().strip()
                if not text:
                    continue
                if text.lower() in ['back', 'exit', 'quit']:
                    print("↩️  Returning to main REPL...")
                    break
                    
                # Make prediction
                result = self._predict_sentiment(text)
                if result:
                    sentiment, confidence = result
                    # Get debugging info
                    oov_fraction = getattr(self, '_last_oov_fraction', 0)
                    raw_prob = getattr(self, '_last_raw_prob', 0)
                    
                    # Show detailed info for edge cases (near decision boundaries)
                    if 0.4 <= raw_prob <= 0.8:
                        print(f"Prediction: {sentiment} ({confidence:.1%} confidence)")
                        print(f"  Debug: Raw prob={raw_prob:.3f}, OOV={oov_fraction:.1%}, Threshold=0.7")
                    else:
                        print(f"Prediction: {sentiment} ({confidence:.1%} confidence)")
                print()
                
            except (KeyboardInterrupt, EOFError):
                print("\nReturning to main REPL...")
                break

    def _interactive_mnist_prediction(self):
        """Interactive MNIST digit prediction"""
        print("🔢 MNIST Digit Classifier - Interactive Mode")
        print("💡 This model recognizes handwritten digits (0-9)")
        print("⚠️  Interactive prediction requires trained model and image input")
        print("📝 Type 'back' to return to main REPL.")
        print()
        
        print("🧠 For MNIST prediction, you would typically:")
        print("  1. Load/capture a handwritten digit image")
        print("  2. Preprocess it to 28x28 grayscale")
        print("  3. Feed it to the trained neural network")
        print()
        print("💡 Try 'generate' to see the complete Python code")
        print("💡 Or 'train' to see the model training process")
        print()
        
        input("Press Enter to return to main REPL...")

    def _quick_train_model(self):
        """🔥 CRITICAL: Train the actual neural network model"""
        try:
            # Clear any existing cached models
            import os
            if os.path.exists('/tmp/trained_model.h5'):
                os.remove('/tmp/trained_model.h5')
            if os.path.exists('/tmp/tokenizer.pkl'):
                os.remove('/tmp/tokenizer.pkl')
            
            # Clear cached model and tokenizer
            self.cached_model = None
            self.cached_tokenizer = None
            
            print("📊 Loading your training data from CSV...")
            
            # Format model for generation
            model_for_generation = {
                'name': self.current_model['name'],
                'sections': {
                    'dataset': self.current_model['dataset'],
                    'architecture': self.current_model['architecture'],
                    'training': self.current_model['training']
                }
            }
            
            # Generate Python code that creates sample data and trains model
            python_code = self._generate_training_code(model_for_generation)
            
            # Save to temp file and execute
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
                tmp.write(python_code)
                self.trained_model_path = tmp.name
            
            # Execute the training and show output
            env = os.environ.copy()
            result = subprocess.run([sys.executable, self.trained_model_path],
                                  capture_output=True,  # Still capture to check for errors
                                  text=True,
                                  env=env)
            
            # Always show the training output
            if result.stdout:
                print("=== TRAINING OUTPUT ===")
                print(result.stdout)
                print("=====================")
                
            if result.returncode == 0:
                return True
            else:
                print(f"Training failed with error:")
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"Training exception: {e}")
            return False

    def _train_general_model(self):
        """Train a general model"""
        try:
            # Format model for generation
            model_for_generation = {
                'name': self.current_model['name'],
                'sections': {
                    'dataset': self.current_model['dataset'],
                    'architecture': self.current_model['architecture'],
                    'training': self.current_model['training']
                }
            }
            
            python_code = generate_python_code(model_for_generation)
            
            # Create temp file and execute
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
                tmp.write(python_code)
                temp_path = tmp.name
            
            # Execute the model
            result = subprocess.run([sys.executable, temp_path], 
                                  capture_output=True, text=True)
            
            # Cleanup
            os.unlink(temp_path)
            
            if result.returncode == 0:
                print("📈 Training output:")
                print(result.stdout[-500:])  # Show last 500 chars
                return True
            else:
                print("❌ Training failed:")
                print(result.stderr[-300:])  # Show last 300 chars of error
                return False
            
        except Exception as e:
            print(f"❌ Error during training: {e}")
            return False

    def _generate_training_code(self, model_definition):
        """Generate Python code that trains the model and saves it"""
        model_name = model_definition["name"]
        sections = model_definition["sections"]
        
        code = [
            "import os",
            "os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings",
            "import warnings",
            "warnings.filterwarnings('ignore')",
            "import tensorflow as tf",
            "from tensorflow.keras import layers, models",
            "import numpy as np",
            "import pickle",
            ""
        ]
        
        # Generate sample data based on model type
        dataset = sections.get('dataset', {})
        source = dataset.get('source', '')
        
        if source == 'language':
            code.extend([
                "# Load your language training data from CSV",
                "import pandas as pd",
                "import os",
                "",
                "# Find the data directory",
                f"data_path = '{os.path.join(os.path.dirname(__file__), '..', 'data', 'language_data.csv')}'",
                "if not os.path.exists(data_path):",
                f"    data_path = '{os.path.join(os.path.dirname(__file__), 'data', 'language_data.csv')}'",
                "if not os.path.exists(data_path):",
                "    raise FileNotFoundError(f'Could not find language_data.csv at {data_path}')",
                "",
                "# Load the CSV data",
                "df = pd.read_csv(data_path)",
                "texts = df['text'].tolist()",
                "",
                "# Convert text labels to numeric",
                "label_map = {'English': 0, 'French': 1, 'Spanish': 2}",
                "labels = [label_map[label] for label in df['label']]",
                "",
                "# DIAGNOSTIC: Check label distribution",
                "import numpy as np",
                "unique_labels, counts = np.unique(labels, return_counts=True)",
                "print(f'Label distribution: {dict(zip(unique_labels, counts))}')",
                "print(f'Sample texts and labels:')",
                "for i in range(min(6, len(texts))):",
                "    print(f'  {texts[i][:30]}... -> {labels[i]} ({list(label_map.keys())[labels[i]]})')",
                "",
                "print(f'Loaded {len(texts)} training samples from real dataset')",
                "print(f'Languages: {set(df[\"label\"])}')",
                "print(f'English: {sum(1 for l in df[\"label\"] if l == \"English\")} samples')",
                "print(f'French: {sum(1 for l in df[\"label\"] if l == \"French\")} samples')",
                "print(f'Spanish: {sum(1 for l in df[\"label\"] if l == \"Spanish\")} samples')",
                "",
                "from tensorflow.keras.preprocessing.text import Tokenizer",
                "from tensorflow.keras.preprocessing.sequence import pad_sequences",
                "from sklearn.model_selection import train_test_split",
                "import re",
                "import unicodedata",
                "",
                "# Fix: Add normalization with contraction handling",
                "def normalise(text):",
                "    # Handle contractions first",
                "    text = text.lower()",
                "    text = text.replace(\"don't\", \"dont\")",
                "    text = text.replace(\"won't\", \"wont\")",
                "    text = text.replace(\"can't\", \"cant\")",
                "    text = text.replace(\"i'm\", \"im\")",
                "    text = text.replace(\"you're\", \"youre\")",
                "    text = text.replace(\"we're\", \"were\")",
                "    text = text.replace(\"they're\", \"theyre\")",
                "    text = text.replace(\"isn't\", \"isnt\")",
                "    text = text.replace(\"wasn't\", \"wasnt\")",
                "    # Strip accents",
                "    text = unicodedata.normalize('NFD', text)",
                "    text = ''.join(ch for ch in text if unicodedata.category(ch) != 'Mn')",
                "    return re.sub(r'[^a-zñáéíóúü ]+', ' ', text)",
                "",
                "# Apply normalization to all training texts",
                "normalized_texts = [normalise(text) for text in texts]",
                "",
                "# Fix 1: Improve tokenizer with larger vocab and lowercase",
                "tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>', lower=True)",
                "tokenizer.fit_on_texts(normalized_texts)",
                "",
                "# DIAGNOSTIC: Check tokenizer learned vocabulary",
                "print(f'Tokenizer vocabulary size: {len(tokenizer.word_index)}')",
                "print(f'Actual vocab kept: {min(5000, len(tokenizer.word_index))}')",
                "print(f'Sample vocabulary: {list(tokenizer.word_index.items())[:10]}')",
                "",
                "# Tokenize the normalized texts",
                "sequences = tokenizer.texts_to_sequences(normalized_texts)",
                "",
                "# Fix 2: Add OOV dropout augmentation to balance exposure",
                "def dropout_oov(seq, rate=0.15):",
                "    import random",
                "    return [tok if tok == 1 or random.random() > rate else 1 for tok in seq]",
                "",
                "# Apply OOV dropout to training sequences",
                "augmented_sequences = [dropout_oov(s) for s in sequences]",
                "padded = pad_sequences(augmented_sequences, maxlen=100, padding='post', truncating='post')",
                "",
                "# DIAGNOSTIC: Check tokenization works",
                "test_samples = ['Hello', 'Bonjour', 'Hola']",
                "test_sequences = tokenizer.texts_to_sequences(test_samples)",
                "print(f'Test tokenization: {dict(zip(test_samples, test_sequences))}')",
                "",
                "# Fix 3: Proper train/validation split with stratification",
                "x_train, x_val, y_train, y_val = train_test_split(",
                "    padded, labels, test_size=0.2, stratify=labels, random_state=42)",
                "",
                "# Fix data types for Keras compatibility - use int32 for embeddings",
                "x_train = np.asarray(x_train, dtype=np.int32)",
                "x_val = np.asarray(x_val, dtype=np.int32)",
                "y_train = np.array(y_train, dtype=np.int32)",
                "y_val = np.array(y_val, dtype=np.int32)",
                "",
                "print(f'Training samples: {len(x_train)}, Validation samples: {len(x_val)}')",
                "print(f'Train label distribution: {np.bincount(y_train)}')",
                "print(f'Val label distribution: {np.bincount(y_val)}')",
                "x_test = x_val  # Use validation set for testing",
                "y_test = y_val",
                ""
            ])
        elif source == 'sentiment' or source == 'custom':
            code.extend([
                "# Load your sentiment training data from CSV",
                "import pandas as pd",
                "import os",
                "",
                "# Find the data directory",
                f"data_path = '{os.path.join(os.path.dirname(__file__), '..', 'data', 'sentiment_data.csv')}'",
                "if not os.path.exists(data_path):",
                f"    data_path = '{os.path.join(os.path.dirname(__file__), 'data', 'sentiment_data.csv')}'",
                "if not os.path.exists(data_path):",
                "    raise FileNotFoundError(f'Could not find sentiment_data.csv at {data_path}')",
                "",
                "# Load the CSV data",
                "df = pd.read_csv(data_path)",
                "texts = df['text'].tolist()",
                "",
                "# Convert text labels to numeric",
                "if 'label' in df.columns:",
                "    label_map = {'negative': 0, 'positive': 1}",
                "    labels = [label_map[label.lower()] for label in df['label']]",
                "else:",
                "    # Handle sentiment column with numeric values",
                "    labels = df['sentiment'].tolist()",
                "",
                "print(f'Loaded {len(texts)} training samples from real dataset')",
                "",
                "from tensorflow.keras.preprocessing.text import Tokenizer",
                "from tensorflow.keras.preprocessing.sequence import pad_sequences",
                "from sklearn.model_selection import train_test_split",
                "import re",
                "import unicodedata",
                "",
                "# Add same normalization function for sentiment",
                "def normalise(text):",
                "    # Handle contractions first",
                "    text = text.lower()",
                "    text = text.replace(\"don't\", \"dont\")",
                "    text = text.replace(\"won't\", \"wont\")",
                "    text = text.replace(\"can't\", \"cant\")",
                "    text = text.replace(\"i'm\", \"im\")",
                "    text = text.replace(\"you're\", \"youre\")",
                "    text = text.replace(\"we're\", \"were\")",
                "    text = text.replace(\"they're\", \"theyre\")",
                "    text = text.replace(\"isn't\", \"isnt\")",
                "    text = text.replace(\"wasn't\", \"wasnt\")",
                "    # Strip accents",
                "    text = unicodedata.normalize('NFD', text)",
                "    text = ''.join(ch for ch in text if unicodedata.category(ch) != 'Mn')",
                "    return re.sub(r'[^a-zñáéíóúü ]+', ' ', text)",
                "",
                "# Apply normalization to all sentiment texts",
                "normalized_texts = [normalise(text) for text in texts]",
                "",
                "# Use larger vocab to capture negation words like 'dont', 'wont'",
                "tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>', lower=True)",
                "tokenizer.fit_on_texts(normalized_texts)",
                "sequences = tokenizer.texts_to_sequences(normalized_texts)",
                "",
                "# No OOV dropout for sentiment - we need the actual words",
                "padded = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')",
                "",
                "# DIAGNOSTIC: Check labels are binary (0/1)",
                "print(f'Label distribution: {np.unique(labels, return_counts=True)}')",
                "print(f'Label range: {min(labels)} to {max(labels)}')",
                "",
                "# DIAGNOSTIC: Check negation tokenization works",
                "print(f'Vocabulary size: {len(tokenizer.word_index)}')",
                "test_sequences = tokenizer.texts_to_sequences(['i love this', 'i dont love this', 'terrible', 'amazing'])",
                "print(f'Test tokenization: {test_sequences}')",
                "# Check if key negation words are in vocabulary",
                "negation_words = ['dont', 'wont', 'cant', 'isnt', 'hate', 'love', 'not']",
                "negation_indices = {word: tokenizer.word_index.get(word, 'OOV') for word in negation_words}",
                "print(f'Negation word indices: {negation_indices}')",
                "",
                "# Proper train/validation split with data type fix",
                "x_train, x_val, y_train, y_val = train_test_split(",
                "    padded, labels, test_size=0.2, stratify=labels, random_state=42)",
                "",
                "# Fix data types for Keras compatibility - use int32 for embeddings",
                "x_train = np.asarray(x_train, dtype=np.int32)",
                "x_val = np.asarray(x_val, dtype=np.int32)",
                "y_train = np.array(y_train, dtype=np.float32)  # float32 for binary labels",
                "y_val = np.array(y_val, dtype=np.float32)",
                "",
                "print(f'Final y_train unique: {np.unique(y_train)}')",
                "print(f'Training samples: {len(x_train)}, Validation: {len(x_val)}')",
                "x_test = x_val",
                "y_test = y_val",
                ""
            ])
        
        # Build model architecture
        code.extend([
            "# Build model architecture",
            "model = models.Sequential()"
        ])
        
        if "architecture" in sections and "layers" in sections["architecture"]:
            for layer in sections["architecture"]["layers"]:
                layer_type = list(layer.keys())[0]
                layer_params = layer[layer_type]
                
                # Fix 4: Add L2 regularization to Dense layers
                if layer_type == "Dense" and layer_params.get("activation") == "relu":
                    layer_params["kernel_regularizer"] = "tf.keras.regularizers.l2(1e-4)"
                
                params_str = ", ".join([f"{k}={repr(v) if not isinstance(v, str) or not v.startswith('tf.') else v}" for k, v in layer_params.items()])
                code.append(f"model.add(layers.{layer_type}({params_str}))")
        
        # Training configuration
        if "training" in sections:
            training = sections["training"]
            optimizer = training.get("optimizer", "adam")
            loss = training.get("loss", "sparse_categorical_crossentropy")
            metrics = training.get("metrics", ["accuracy"])
            epochs = min(training.get("epochs", 30), 30)  # Slightly more epochs but with early stopping
            
            code.extend([
                "",
                "# Fix 5: Add early stopping callback with higher patience for sentiment",
                "from tensorflow.keras.callbacks import EarlyStopping",
                "early_stopping = EarlyStopping(",
                "    monitor='val_loss',",
                "    patience=10,",
                "    restore_best_weights=True,",
                "    verbose=1",
                ")",
                "",
                "# Compile and train model",
                f"model.compile(optimizer='{optimizer}', loss='{loss}', metrics={metrics})",
                "",
                "# DIAGNOSTIC: Check training actually works",
                f"print('Before training - evaluating on training data:')",
                f"initial_loss, initial_acc = model.evaluate(x_train, y_train, verbose=0)",
                f"print(f'Initial loss: {{initial_loss:.4f}}, Initial accuracy: {{initial_acc:.4f}}')",
                "",
                f"print('Training for up to {epochs} epochs with early stopping...')",
                f"history = model.fit(",
                f"    x_train, y_train,",
                f"    validation_data=(x_val, y_val),",
                f"    epochs={epochs},",
                f"    callbacks=[early_stopping],",
                f"    verbose=1",
                f")",
                "",
                "# Fix 7: Validation accuracy tracking and reporting",
                f"train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)",
                f"val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)",
                f"print(f'\\nFINAL RESULTS:')",
                f"print(f'Train accuracy: {{train_acc:.4f}} | Validation accuracy: {{val_acc:.4f}}')",
                f"print(f'Train loss: {{train_loss:.4f}} | Validation loss: {{val_loss:.4f}}')",
                "",
                "# Check for overfitting",
                "if train_acc - val_acc > 0.15:",
                "    print('WARNING: Large gap between train and validation accuracy - possible overfitting!')",
                "elif val_acc < 0.6:",
                "    print('WARNING: Low validation accuracy - model may need more data or tuning')",
                "else:",
                "    print('✅ Good generalization: train and validation accuracies are close')",
                "",
                "# DIAGNOSTIC: Check weight statistics",
                "embed_weights = model.get_layer('embedding').get_weights()[0]",
                "dense_weights = model.layers[-1].get_weights()[0]", 
                "print(f'Embedding weights std: {embed_weights.std():.6f}')",
                "print(f'Dense weights std: {dense_weights.std():.6f}')",
                "",
                "# Save model and tokenizer",
                "model.save('/tmp/trained_model.h5', save_format='h5')",
                "with open('/tmp/tokenizer.pkl', 'wb') as f:",
                "    pickle.dump(tokenizer, f)",
                "",
                "print('Training complete!')"
            ])
        
        return "\n".join(code)

    def _predict_language(self, text):
        """ CRITICAL: Make a language prediction using trained neural network"""
        try:
            # Try to use trained model first
            import os
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
            import warnings
            warnings.filterwarnings('ignore')
            import tensorflow as tf
            import pickle
            
            # Load model and tokenizer only if not cached
            if self.cached_model is None or self.cached_tokenizer is None:
                if os.path.exists('/tmp/trained_model.h5') and os.path.exists('/tmp/tokenizer.pkl'):
                    self.cached_model = tf.keras.models.load_model('/tmp/trained_model.h5', compile=False)
                    self.cached_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                    
                    with open('/tmp/tokenizer.pkl', 'rb') as f:
                        self.cached_tokenizer = pickle.load(f)
                else:
                    print("No trained model found. Please run 'train' first to train the neural network.")
                    return None
            
            if self.cached_model is not None and self.cached_tokenizer is not None:
                
                # Apply same normalization as training
                import re, unicodedata
                def normalise(text):
                    text = unicodedata.normalize('NFD', text.lower())
                    text = ''.join(ch for ch in text if unicodedata.category(ch) != 'Mn')
                    return re.sub(r'[^a-zñáéíóúü ]+', ' ', text)
                
                normalized_text = normalise(text)
                
                # Preprocess input text
                sequence = self.cached_tokenizer.texts_to_sequences([normalized_text])
                from tensorflow.keras.preprocessing.sequence import pad_sequences
                padded = pad_sequences(sequence, maxlen=100, padding='post', truncating='post')
                
                # Calculate OOV fraction for adaptive confidence
                seq_oov_fraction = (padded[0] == 1).sum() / len(padded[0])
                
                # Make prediction
                prediction = self.cached_model.predict(padded, verbose=0)[0]
                
                # Get the predicted class and confidence
                predicted_class = prediction.argmax()
                confidence = prediction.max()
                
                # Store OOV fraction for confidence threshold
                self._last_oov_fraction = seq_oov_fraction
                
                languages = ['English', 'French', 'Spanish']
                return languages[predicted_class], confidence
            else:
                # No trained model available - train first
                print("No trained model found. Please run 'train' first to train the neural network.")
                return None
                
        except Exception as e:
            # Training failed or model error
            print(f"Neural network error: {str(e)}")
            print("Try running 'train' again or check your data files.")
            return None

    def _predict_sentiment(self, text):
        """🔥 CRITICAL: Make sentiment prediction using trained neural network"""
        try:
            # Try to use trained model first
            import os
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
            import warnings
            warnings.filterwarnings('ignore')
            import tensorflow as tf
            import pickle
            
            # Load model and tokenizer only if not cached
            if self.cached_model is None or self.cached_tokenizer is None:
                if os.path.exists('/tmp/trained_model.h5') and os.path.exists('/tmp/tokenizer.pkl'):
                    print("Loading model into memory...")
                    self.cached_model = tf.keras.models.load_model('/tmp/trained_model.h5', compile=False)
                    self.cached_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                    with open('/tmp/tokenizer.pkl', 'rb') as f:
                        self.cached_tokenizer = pickle.load(f)
                else:
                    print("No trained model found. Please run 'train' first to train the neural network.")
                    return None
            
            if self.cached_model is not None and self.cached_tokenizer is not None:
                # Apply same normalization as training
                import re, unicodedata
                def normalise(text):
                    text = unicodedata.normalize('NFD', text.lower())
                    text = ''.join(ch for ch in text if unicodedata.category(ch) != 'Mn')
                    return re.sub(r'[^a-zñáéíóúü ]+', ' ', text)
                
                normalized_text = normalise(text)
                
                # Preprocess input text
                sequence = self.cached_tokenizer.texts_to_sequences([normalized_text])
                from tensorflow.keras.preprocessing.sequence import pad_sequences
                padded = pad_sequences(sequence, maxlen=100, padding='post', truncating='post')
                
                # Calculate OOV fraction for adaptive confidence
                seq_oov_fraction = (padded[0] == 1).sum() / len(padded[0])
                
                # Make prediction
                prediction = self.cached_model.predict(padded, verbose=0)[0]
                
                # Fix: Handle sigmoid output correctly
                prob = float(prediction[0])  # Extract scalar probability [0, 1]
                
                # Raised threshold to reduce false positives
                sentiment = 'Positive' if prob >= 0.7 else 'Negative'
                confidence = prob if sentiment == 'Positive' else 1 - prob
                
                # Store OOV fraction and raw probability for debugging
                self._last_oov_fraction = seq_oov_fraction
                self._last_raw_prob = prob
                
                return sentiment, confidence
            else:
                # No trained model available - train first
                print("No trained model found. Please run 'train' first to train the neural network.")
                return None
                
        except Exception as e:
            # Training failed or model error
            print(f"Neural network error: {str(e)}")
            print("Try running 'train' again or check your data files.")
            return None

    def show_history(self):
        """Show command history"""
        print("📝 Command History:")
        for i, cmd in enumerate(self.history[-10:], 1):  # Show last 10 commands
            print(f"  {i}. {cmd}")
        print()

def main():
    repl = ModelScriptREPL()
    repl.start()

if __name__ == "__main__":
    main()