import os
import numpy as np
import json
from collections import defaultdict

def load_tonic(ctonic_path):
    """Load tonic frequency from .ctonic.txt file (handles scalar/array)"""
    tonic_data = np.loadtxt(ctonic_path)
    return tonic_data.item() if tonic_data.ndim == 0 else tonic_data[0]

def load_pitch_data(pitch_path):
    """Load time-stamped pitch values from .pitch.txt"""
    if not os.path.exists(pitch_path):
      print(f"Missing pitch file: {pitch_path}")
      return None
    return np.loadtxt(pitch_path)

def load_sections(sections_path):
    """Load sections from comma-separated .sections-manual-p.txt files"""
    sections = []
    if os.path.exists(sections_path):
        try:
            with open(sections_path, "r", encoding="utf-8") as f:  # Specify UTF-8 encoding here
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # Split by commas
                    parts = line.split(",")
                    if len(parts) < 4:  # Example line: "0.0,1,129.869,Ālāp"
                        print(f"Skipping invalid line: {line}")
                        continue

                    # Extract start time and label
                    start = float(parts[0])
                    end = float(parts[2])  # Third column is end time
                    label = parts[3].strip().lower().replace(" ", "_").replace("(", "").replace(")", "")  # Normalize labels (e.g., "Ālāp" → "alap")

                    sections.append((start, end, label))
        except Exception as e:
            print(f"Error reading {sections_path}: {str(e)}")
    else:
        print(f"Sections file missing: {sections_path}")
    return sections
def hz_to_svara(freq, tonic_hz):
    """Convert frequency to Indian svara (Sa, Re, Ga, etc.)"""
    if freq == 0:
        return None
    cents = 1200 * np.log2(freq / tonic_hz)
    svaras = {0: 'Sa', 200: 'Re', 300: 'Ga', 500: 'Ma', 700: 'Pa', 900: 'Dha', 1100: 'Ni'}
    nearest = min(svaras.keys(), key=lambda x: abs(x - cents))
    return svaras[nearest]

def preprocess_raag(raag_folder):
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

        if pitch_data is None:
          return []

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
def extract_raag_names(folder_name):
    """Extract raag names from folder names."""
    import re
    # Improved regex to handle more variations in folder names
    folder_name = re.sub(r'\bby\b.*|\b&\b.*|\(.*?\)|\[.*?\]', '', folder_name, flags=re.IGNORECASE)
    raags = []
    for part in re.split(r',|&|/|_', folder_name):
        part = part.strip()
        if part and len(part) > 3:
            raags.append(part.title().strip())
    return list(set(raags))

def create_sequences(all_notes, sequence_length, tokenizer):
    """Create input sequences and target tokens for training."""
    token_ids = [tokenizer.texts_to_sequences([note])[0][0] for note in all_notes]
    X, y = [], []
    for i in range(len(token_ids) - sequence_length):
        X.append(token_ids[i:i+sequence_length])
        y.append(token_ids[i+sequence_length])
    return np.array(X), np.array(y)
def generate_raag_labels(root_path, X, all_notes, raag_id, num_raags):
    """
    Generates raag labels for each note sequence in X.

    Args:
        root_path (str): Path to the root directory of the dataset.
        X (np.ndarray): Input note sequences.
        all_notes (list): List of all notes in the dataset.
        raag_id (dict): Mapping of raag names to numerical IDs.
        num_raags (int): Number of unique raags.

    Returns:
        np.ndarray: Array of raag labels, with shape matching X.
    """

    raag_labels = []
    note_index = 0  # Track current position in all_notes

    for root, dirs, files in os.walk(root_path):
        for dir_name in dirs:
            if "Raag" in dir_name:
                raag_names = extract_raag_names(dir_name)
                if raag_names:  # Ensure raag_names is not empty
                    raag_id_value = raag_id.get(raag_names[0])  # Get raag ID
                    # Ensure raag_id_value is valid
                    raag_id_value = 0 if raag_id_value is None or raag_id_value >= num_raags else raag_id_value

                    # Determine the number of notes for this raag
                    num_notes_for_raag = 0
                    while note_index < len(all_notes) and all_notes[note_index] in dir_name:
                        num_notes_for_raag += 1
                        note_index += 1

                    # Extend raag_labels with the appropriate ID for these notes
                    raag_labels.extend([raag_id_value] * num_notes_for_raag)

    # Ensure raag_labels has the same length as X (handle potential discrepancies)
    raag_labels = raag_labels[:len(X)]  # Trim if longer
    raag_labels = np.pad(raag_labels, (0, len(X) - len(raag_labels)), 'constant')  # Pad if shorter

    return np.array(raag_labels)