import streamlit as st
import subprocess
import os
import tempfile
import sys
import pickle
import tensorflow as tf
import numpy as np
import pretty_midi
from src.models.music_utils import tokens_to_midi  # Import tokens_to_midi
from src.models.model_builder import create_model  # Import model builder
from src.models.data_utils import create_tokenizer, load_and_preprocess_data, extract_all_notes, create_raag_id_mapping
from src.models.music_utils import generate_random_seed, generate_music_with_tonic
import yaml

# Define a path to a dummy requirements.txt
requirements_file = "streamlit_requirements.txt"
with open(requirements_file, 'w') as f:
    f.write("streamlit==1.31.0\npretty_midi==0.2.9\ntensorflow==2.15.0\nkeras==2.15.0")

# Function to install requirements
def install_packages(requirements_file):
    try:
        with open(requirements_file, 'r') as f:
            packages = [line.strip() for line in f if line.strip()] # Remove pickle package from the list.
        subprocess.run([sys.executable, "-m", "pip", "install", *packages, "--no-cache-dir"], check=True, capture_output=True)
        st.success("Successfully installed requirements.")
    except subprocess.CalledProcessError as e:
        st.error(f"Error installing requirements: {e.stderr.decode()}")

# Install requirements
install_packages(requirements_file)

#set the strategy
strategy = tf.distribute.get_strategy()

# Load model and tokenizer
model = None
tokenizer = None
vocab_size = None
num_raags = None
sequence_length = None
raag_id_dict = None

# Load the model with custom layers (if you defined any)
def load_model(repo_dir):
    """Loads the model from the specified path."""
    global vocab_size, num_raags, sequence_length #use the globals
    model_name = "my_model"  # Update if you change the model name
    model_path = os.path.join(repo_dir, "models", f"{model_name}.h5")
    
    try:
        with strategy.scope():
            # First, we need to know the shape
            config_file = os.path.join(repo_dir, "config.yaml")
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
            sequence_length = config['sequence_length']
            
            # Now load the model
            model = create_model(vocab_size, num_raags, sequence_length, strategy)
            model.load_weights(model_path)

        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        st.error(f"Error loading model: {e}")
        return None

# Load tokenizer
def load_tokenizer(repo_dir):
    """Loads the tokenizer from the specified path."""
    tokenizer_name = "my_tokenizer"  # Update if you change the tokenizer name
    tokenizer_path = os.path.join(repo_dir, "tokenizers", f"{tokenizer_name}.pickle")
    try:
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
        print("Tokenizer loaded successfully")
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        st.error(f"Error loading tokenizer: {e}")
        return None

def load_raag_data(repo_dir, data_path):
    global raag_id_dict, num_raags
    all_output = load_and_preprocess_data(repo_dir, data_path)
    all_notes, all_output_filtered = extract_all_notes(all_output)
    raag_id_dict, num_raags = create_raag_id_mapping(all_output_filtered)
    print("Raag data loaded")

print("Starting the App")

# Determine paths based on the environment (Colab or local)
if os.environ.get("COLAB_GPU", "FALSE") == "TRUE":
    repo_dir = "/content/drive/MyDrive/music_generation_repo"
    data_path = "/content/drive/MyDrive/"
    print(f"Running on Colab. repo_dir: {repo_dir}")
    print(f"Running on Colab. data_path: {data_path}")
else:
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(repo_dir) # go up one level
    data_path = os.path.dirname(repo_dir)
    print(f"Running locally. repo_dir: {repo_dir}")
    print(f"Running locally. data_path: {data_path}")

# Load model and tokenizer if not already loaded
# Load tokenizer
tokenizer = load_tokenizer(repo_dir)
if tokenizer is not None:
    vocab_size = len(tokenizer.word_index) +1
    print(f"vocab_size: {vocab_size}")
    load_raag_data(repo_dir, data_path) #load data about raag.
    model = load_model(repo_dir) # load model after having the raag data.

st.title("Indian Raga Music Generator")

if model is None or tokenizer is None or num_raags is None or vocab_size is None or raag_id_dict is None:
    st.error("There was an error loading one or more of the components. Please check the console for details.")
else:
    # Select raag
    selected_raag = st.selectbox('Select a Raag', list(raag_id_dict.keys()))
    raag_id = raag_id_dict[selected_raag]

    if st.button("Generate Music"):
      print("Model and Tokenizer are not None")
      #seed_sequence = [tokenizer.word_index["Sa"]] * 50  # Example seed
      seed_sequence = generate_random_seed(tokenizer, 50) #new seed
      print("Before generate_music")
      generated_tokens = generate_music_with_tonic(model, seed_sequence, raag_id, tokenizer, max_length=200, temperature=1.8, top_k=9, strategy=strategy, vocab_size=vocab_size, sequence_length=50)
      print("After generate_music")


      # Map tokens to svaras (include only musical notes)
      valid_notes = {"Sa", "Re", "Ga", "Ma", "Pa", "Dha", "Ni"}
      generated_notes = [
          tokenizer.index_word.get(token, '<UNK>')
          for token in generated_tokens
          if tokenizer.index_word.get(token, '<UNK>') in valid_notes
      ]
      print("After mapping tokens to notes")

      try:
        midi_path = tokens_to_midi(generated_notes)
        print("MIDI file path:", midi_path)
        st.audio(midi_path)
        print("Successfully displayed audio player")
      except Exception as e:
        print(f"Error converting to MIDI or displaying audio: {e}")
        st.error(f"Error converting to MIDI or displaying audio: {e}")

