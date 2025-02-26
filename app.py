import streamlit as st
import subprocess
import os
import tempfile
import sys
import pickle
import tensorflow as tf
from src.models.music_utils import tokens_to_midi  # Import tokens_to_midi
#from music_utils import tokens_to_midi  # Import tokens_to_midi
# Define a path to a dummy requirements.txt
requirements_file = "streamlit_requirements.txt"
with open(requirements_file, 'w') as f:
    f.write("streamlit==1.31.0\npretty_midi\ntensorflow\nkeras-nlp") # added a version number for streamlit

# Function to install requirements
def install_packages(requirements_file):
    try:
        with open(requirements_file, 'r') as f:
            packages = [line.strip() for line in f if line.strip() and "pickle" not in line.lower()] # Remove pickle package from the list.
        subprocess.run([sys.executable, "-m", "pip", "install", *packages], check=True, capture_output=True)
        st.success("Successfully installed requirements.")
    except subprocess.CalledProcessError as e:
        st.error(f"Error installing requirements: {e.stderr.decode()}")

# Install requirements
install_packages(requirements_file)

# Load model and tokenizer
model = None
tokenizer = None

print("Before model loading")
try:
    # Load the model with custom layers (if you defined any)
    model = tf.keras.models.load_model(
        "indianraga_model.keras",
        custom_objects={
            "RaagConditioning": tf.keras.layers.Layer,
            "TransformerBlock": tf.keras.layers.Layer,
            "MultiHeadAttention": tf.keras.layers.Layer
        }
    )
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    st.error(f"Error loading model: {e}")
print("After model loading")

print("Before tokenizer loading")
try:
    with open("ragatokenzier.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    print("Tokenizer loaded successfully")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    st.error(f"Error loading tokenizer: {e}")
print("After tokenizer loading")

st.title("Indian Raga Music Generator")

if st.button("Generate Music"):
    if model is not None and tokenizer is not None:
      print("Model and Tokenizer are not None")
      seed_sequence = [tokenizer.word_index["Sa"]] * 50  # Example seed
      raag_id = 0  # Choose a raag ID
      print("Before generate_music")

      def generate_music(model, seed_sequence, raag_id, max_length=200, temperature=1.8, top_k=9):
          """Generates music using the model, incorporating temperature and top-k sampling."""
          generated = seed_sequence.copy()  # Initialize generated sequence
          for _ in range(max_length):
            # Trim sequence to the last `sequence_length` tokens
            input_seq = tf.constant([generated[-50:]], dtype=tf.int32) #convert list to tensor with int32 dtype

            # Reshape input_seq to match the expected input shape of the model
            #input_seq = input_seq.reshape(1, 50)  # Reshape to (1, sequence_length)
            raag_input = tf.constant([[raag_id]], dtype=tf.int32)

            # Predict next token probabilities
            pred = model([input_seq, raag_input], training=False)[0]


            # Apply temperature scaling
            pred = pred / temperature
            pred = tf.nn.softmax(pred).numpy()  # Softmax


            # Top-k sampling
            top_k_indices = np.argsort(pred)[-top_k:]
            top_k_probs = pred[top_k_indices]
            top_k_probs = top_k_probs / np.sum(top_k_probs)  # Renormalize

            # Sample from top-k
            next_token = np.random.choice(top_k_indices, p=top_k_probs)
            generated.append(next_token)

          return generated


      generated_tokens = generate_music(model, seed_sequence, raag_id)
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

    else:
        st.error("Model or Tokenizer not loaded successfully")
        print("Model or Tokenizer not loaded successfully")