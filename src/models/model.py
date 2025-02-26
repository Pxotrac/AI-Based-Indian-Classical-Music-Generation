import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from data_utils import load_tonic, load_pitch_data, load_sections, hz_to_svara, preprocess_raag, extract_raag_names, generate_raag_labels, create_sequences
from music_utils import generate_music, notes_to_midi, display_audio
from model_builder import create_model  # Import the model definition

# --- File Paths ---
root_path = r"E:\Projects\AI-Based-Indian-Classical-Music-Generation\data\hindustani"
tokenizer_path = "ragatokenzier.pkl"
model_checkpoint_path = "model_checkpoint_{epoch:02d}.h5" # Path to save model checkpoints
# Example raag_folder, change as necessary, this is now only used to load the tokenizer
raag_folder = r"E:\Projects\AI-Based-Indian-Classical-Music-Generation\data\hindustani\Anaahata by Milind Malshe\Raag Basanti Kedar"


# --- Load tokenizer ---
try:
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    print("Tokenizer loaded successfully!")
    print("Vocabulary size:", len(tokenizer.word_index))

except FileNotFoundError as e:
    print(f"Error loading data files: {e}. Please make sure files are present.")
except Exception as e:
    print(f"An unexpected error occured while loading the tokenizer: {e}")


# --- Load preprocessed sequence from JSON or by preprocessing ---
# This part will now be used to get a vocab, raag_id_dict and all_notes.
output = preprocess_raag(raag_folder)

# Generate all_notes for generating raag_labels
all_notes = []
for entry in output:
    if "svara" in entry:
        all_notes.append(entry["svara"])
    elif "section" in entry:
        all_notes.append(entry["section"])

# --- Get Raag IDs ---
# Extract all raag names
all_raag_names = set()
for root, dirs, files in os.walk(root_path):
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

# --- Collect Data from All Raags ---
all_data = [] # Initialize an empty list to store all the data
for root, dirs, files in os.walk(root_path):
    for dir_name in dirs:
        if "Raag" in dir_name:
            raag_path = os.path.join(root, dir_name)
            print(f"Preprocessing raag: {raag_path}")
            raag_data = preprocess_raag(raag_path)
            if raag_data:
              all_data.extend(raag_data) # Add the preprocessed data

# --- Generate all_notes from the all_data---
all_notes = []
for entry in all_data:
    if "svara" in entry:
        all_notes.append(entry["svara"])
    elif "section" in entry:
        all_notes.append(entry["section"])


# --- Create sequences (X, y) for training ---
sequence_length = 50  # Context window
X, y = create_sequences(all_notes, sequence_length, tokenizer)

# --- Model Definition ---
# Generate raag labels
raag_labels = generate_raag_labels(root_path, X, all_notes, raag_id_dict, num_raags)

# Get vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print("Total vocabulary size:", vocab_size)
model = create_model(vocab_size, num_raags, sequence_length) # Create the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()


# --- Training ---
# Define the EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Define the ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath=model_checkpoint_path,
    save_freq='epoch',
    monitor='val_loss',
    save_best_only=True
)

# Reshape raag_labels to (num_samples, 1)
raag_labels = raag_labels.reshape(-1, 1)

# Train the model with the callbacks
history = model.fit(
    x=[X, raag_labels],
    y=y,
    batch_size=64,
    epochs=100,
    validation_split=0.2,
    callbacks=[early_stopping, checkpoint_callback]
)

# --- Example Usage - Music Generation ---
# Example usage
raag_id_value = raag_id_dict.get('Basanti Kedar', 0)  # Ensure this ID is within 0-57

# Seed with a short melody instead of all "Sa"
seed = [
    tokenizer.word_index["Sa"], tokenizer.word_index["Re"], tokenizer.word_index["Ga"],
    tokenizer.word_index["Ma"], tokenizer.word_index["Pa"]
] * 10  # Repeat to fill 50 tokens
seed = seed[:sequence_length]  # Trim to exact length

generated_tokens = generate_music(model, seed, raag_id_value, vocab_size, sequence_length, tokenizer, temperature=1.8, top_k=9)

# Map tokens to svaras (include only musical notes)
valid_notes = {"Sa", "Re", "Ga", "Ma", "Pa", "Dha", "Ni"}
generated_notes = [
    tokenizer.index_word.get(token, '<UNK>')
    for token in generated_tokens
    if tokenizer.index_word.get(token, '<UNK>') in valid_notes
]

midi_path = notes_to_midi(generated_notes)
display_audio(midi_path)