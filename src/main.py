!pip install pyyaml
import os
import json
import yaml
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# --- Mount the second drive ---
# Add the following code block BEFORE loading config.yaml
from google.colab import auth
auth.authenticate_user()

from googleapiclient.discovery import build
drive_service = build('drive', 'v3')

# Replace with the folder ID of the shared folder in the second drive
folder_id = "1VZPzVS_M-Y-FEWcouV7dTKI2Alpl6v4F" # Replace with your ID

results = drive_service.files().list(q=f"'{folder_id}' in parents", fields="files(id, name)").execute()

items = results.get('files', [])
if not items:
    print('No files found in the specified folder')
else:
    print(f"Found {len(items)} files in the specified folder")
    print(f"Example: {items[0]['name']}")


# --- Load Config File ---
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# --- Extract Config Variables ---
dataset_path = config["dataset_path"]
sample_rate = config["sample_rate"]
sequence_length = config.get("sequence_length", 50) # Added to config file
preprocessed_sequence_path = "preprocessed_sequence.json"
tokenizer_path = "ragatokenzier.pkl"
# Define all functions from data_utils.py in main.py, as the module is difficult to work with
def load_tonic(ctonic_path):
    """Load tonic frequency from .ctonic.txt file (handles scalar/array)"""
    tonic_data = np.loadtxt(ctonic_path)
    return tonic_data.item() if tonic_data.ndim == 0 else tonic_data[0]

def load_pitch_data(pitch_path):
    """Load time-stamped pitch values from .pitch.txt"""
    return np.loadtxt(pitch_path)

def load_sections(sections_path):
    sections = []
    if os.path.exists(sections_path):
        try:
            with open(sections_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(",")
                    if len(parts) < 4:
                        continue
                    start = float(parts[0])
                    end = float(parts[2])
                    # Remove non-ASCII characters and normalize
                    label = parts[3].encode("ascii", "ignore").decode().strip().lower()
                    label = "_".join(label.split())  # Replace spaces with underscores
                    sections.append((start, end, label))
        except Exception as e:
            print(f"Error: {str(e)}")
    return sections

def hz_to_svara(freq, tonic_hz):
    """Convert frequency to Indian svara (Sa, Re, Ga, etc.)"""
    if freq == 0:
        return None
    cents = 1200 * np.log2(freq / tonic_hz)
    svaras = {0: 'Sa', 200: 'Re', 300: 'Ga', 500: 'Ma', 700: 'Pa', 900: 'Dha', 1100: 'Ni'}
    nearest = min(svaras.keys(), key=lambda x: abs(x - cents))
    return svaras[nearest]

def extract_raag_names(folder_name):
    """Extract raag names from folder names."""
    import re
    folder_name = re.sub(r'\bby\b.*|\b&\b.*|\(.*?\)|\[.*?\]', '', folder_name, flags=re.IGNORECASE)
    raags = []
    for part in re.split(r',|&|/|_', folder_name):
        part = part.strip()
        if part and len(part) > 3:
            raags.append(part.title().strip())
    return list(set(raags))

def preprocess_raag(raag_folder, dataset_path):
    """Full preprocessing pipeline with corrected section file paths"""
    try:
        base_name = os.path.basename(raag_folder)
        ctonic_path = os.path.join(raag_folder, f"{base_name}.ctonic.txt")
        pitch_path = os.path.join(raag_folder, f"{base_name}.pitch.txt")

        # Corrected section file path (matches your dataset's naming convention)
        sections_path = os.path.join(raag_folder, f"{base_name}.sections-manual-p.txt")  # <-- FIX

        # Load data
        tonic_hz = load_tonic(ctonic_path)
        pitch_data = load_pitch_data(pitch_path)
        sections = load_sections(sections_path)  # Now loads from .sections-manual-p.txt

        # Convert to symbolic representation
        note_sequence = []
        for t, freq in pitch_data:
            svara = hz_to_svara(freq, tonic_hz)
            if svara:
                note_sequence.append({"time": t, "svara": svara, "duration": 0.1})

        # Add sections to the sequence
        for start, end, label in sections:
            note_sequence.append({
                "time": start,
                "section": label,  # Ensure this key exists
                "type": "section_start"
            })

        return note_sequence
    except Exception as e:
        print(f"Error processing {raag_folder}: {str(e)}")
        return []

def create_sequences(all_notes, sequence_length, tokenizer):
    """Create input sequences and corresponding output tokens."""
    token_ids = [tokenizer.texts_to_sequences([note])[0][0] for note in all_notes]
    X, y = [], []
    for i in range(len(token_ids) - sequence_length):
        X.append(token_ids[i:i+sequence_length])
        y.append(token_ids[i+sequence_length])
    return np.array(X), np.array(y)

# --- Data Preprocessing ---
# Example raag folder
raag_folder = os.path.join(dataset_path, "Anaahata by Milind Malshe", "Raag Basanti Kedar")  # Example raag_folder, change as necessary

output = preprocess_raag(raag_folder, dataset_path) # Preprocess the data

# Generate all_notes for tokenization
all_notes = []
for entry in output:
    if "svara" in entry:
        all_notes.append(entry["svara"])
    elif "section" in entry:
        all_notes.append(entry["section"])

# --- Tokenizer Configuration ---
tokenizer = Tokenizer(
    filters="",
    lower=False,
    split=None,  # No splitting
    oov_token="<UNK>"
)
tokenizer.fit_on_texts(all_notes)
vocab_size = len(tokenizer.word_index) + 1
print("Token Mapping:", tokenizer.word_index)

# --- Save Tokenizer ---
try:
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)
    print("Tokenizer saved successfully!")
except Exception as e:
    print(f"An unexpected error occured while saving the tokenizer: {e}")

X, y = create_sequences(all_notes, sequence_length, tokenizer)

# Print shapes of X, y and number of samples
print("X shape:", X.shape)
print("y shape:", y.shape)

# --- Get Raag IDs ---
# Extract all raag names
all_raag_names = set()
for root, dirs, files in os.walk(dataset_path):
    for dir_name in dirs:
        if "Raag" in dir_name:
            extracted_raags = extract_raag_names(dir_name)
            all_raag_names.update(extracted_raags)
# Create raag_id mapping
unique_raags = sorted(list(all_raag_names))
raag_id_dict = {raag: idx for idx, raag in enumerate(unique_raags)}

# Print or save raag_id for later use
print("raag_id mapping:", raag_id_dict)
# Count unique raags
num_raags = len(all_raag_names)
print("Number of unique raags:", num_raags)


import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import Input, Embedding, Dense
from tensorflow.keras.models import Model
import keras_nlp

# Custom Layers ----------------------------------------------------------------
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention = tf.nn.softmax(
            (tf.matmul(q, k, transpose_b=True)) / tf.math.sqrt(tf.cast(self.depth, tf.float32))
        )
        output = tf.matmul(scaled_attention, v)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))
        return self.dense(output)

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):  # Add **kwargs
        super().__init__(**kwargs)  # Pass **kwargs to super().__init__
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(d_model, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        attn_output = self.attn(inputs, inputs, inputs)  # Self-attention
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + ffn_output)

class RaagConditioning(tf.keras.layers.Layer):
    def __init__(self, sequence_length, **kwargs):  # Add **kwargs
        super().__init__(**kwargs)  # Pass **kwargs to super().__init__
        self.sequence_length = sequence_length

    def call(self, raag_embed):
        return tf.tile(raag_embed, [1, self.sequence_length, 1])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.sequence_length, input_shape[2])

# Model Definition
num_raags = len(all_raag_names) # Use the number of raags we have discovered
# Inputs
note_input = Input(shape=(sequence_length,), dtype=tf.int32, name="note_input")  # Specify dtype
raag_input = Input(shape=(1,), dtype=tf.int32, name="raag_input")

note_embed = Embedding(vocab_size, 64)(note_input)
raag_embed = Embedding(num_raags, 64)(raag_input)  # Now matches your 58 raags
combined = note_embed + raag_embed

# Transformer (using keras_nlp)
transformer = keras_nlp.layers.TransformerEncoder(
    intermediate_dim=128,  # Adjust if needed
    num_heads=8,
    dropout=0.2
)(combined)

# Output
output = Dense(vocab_size, activation="softmax")(transformer[:, -1, :])

model = Model(inputs=[note_input, raag_input], outputs=output)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
model.summary()