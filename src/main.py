import os
import json
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from data_utils import load_tonic, load_pitch_data, load_sections, hz_to_svara, preprocess_raag, extract_raag_names, create_sequences

# --- File Paths ---
root_path = "/content/drive/MyDrive/hindustani/hindustani"
preprocessed_sequence_path = r"E:\Projects\AI-Based-Indian-Classical-Music-Generation\preprocessed_sequence.json"
tokenizer_path = "ragatokenzier.pkl"
raag_folder = r"E:\Projects\AI-Based-Indian-Classical-Music-Generation\data\hindustani\Anaahata by Milind Malshe\Raag Basanti Kedar" #Example raag_folder, change as necessary

# --- Load preprocessed sequence from JSON or by preprocessing ---
try:
    with open(preprocessed_sequence_path, "r") as f:
        output = json.load(f)
    print("Preprocessed sequence loaded successfully")
except FileNotFoundError:
    print(f"JSON file not found. Generating from scratch")
    output = preprocess_raag(raag_folder)

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

sequence_length = 50
X, y = create_sequences(all_notes, sequence_length, tokenizer)

# Print shapes of X, y and number of samples
print("X shape:", X.shape)
print("y shape:", y.shape)

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