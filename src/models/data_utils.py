import os
import logging
import numpy as np
import pretty_midi
import re
from collections import Counter
import yaml
import tensorflow as tf
from tqdm import tqdm  # Import tqdm for progress bars

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_tonic(filename):
    """Loads the tonic (Sa) frequency from the filename."""
    match = re.search(r"_([0-9.]+)hz_", filename)
    if match:
        return float(match.group(1))
    return None

def load_pitch_data(midi_data):
    """Extracts pitch information from pretty_midi.PrettyMIDI object."""
    pitches = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            pitches.append(note.pitch)
    return pitches

def load_sections(filename):
    """Loads section markers (if any) from the filename."""
    sections = []
    match = re.search(r"_(.*)_", filename)  # Assuming section markers are between underscores
    if match:
        sections = match.group(1).split("|")  # Assuming sections are separated by pipes
    return sections
        
def hz_to_svara(frequency_hz, tonic_hz):
    """Converts frequency in Hz to a svara string."""
    if not frequency_hz or frequency_hz == 0 or not tonic_hz or tonic_hz == 0 :
        return None
    
    ratios = {
        "Sa": 1,
        "Re": 9/8,
        "Ga": 5/4,
        "Ma": 4/3,
        "Pa": 3/2,
        "Dha": 5/3,
        "Ni": 15/8
    }
    
    
    if frequency_hz is None:
      return None
    
    
    svara_mapping = {}
    for svara, ratio in ratios.items():
        svara_freq = tonic_hz * ratio
        svara_mapping[svara] = svara_freq
    
    closest_svara = None
    min_diff = float('inf')
    
    for svara, svara_freq in svara_mapping.items():
      diff = abs(frequency_hz - svara_freq)
      if diff < min_diff:
          min_diff = diff
          closest_svara = svara

    if closest_svara is not None:
        return closest_svara
    else:
      return None

def preprocess_raag(raag_dir, sa_file, pitch_file, sections_file):
    """Preprocesses the data for a given raag."""
    tonic_path = os.path.join(raag_dir, sa_file)
    pitch_path = os.path.join(raag_dir, pitch_file)
    sections_path = os.path.join(raag_dir, sections_file)

    tonic_hz = load_tonic(tonic_path)
    if tonic_hz is None:
        logging.error(f"Failed to load tonic for {raag_dir}")
        return None

    # Assuming you have functions to load pitch data and sections:
    pitch_data = load_pitch_data(pitch_path, tonic_hz)  # Adjust if necessary
    sections = load_sections(sections_path)

    if pitch_data is None or sections is None:
        logging.error(f"Failed to load data for {raag_dir}")
        return None

    if len(sections) != len(pitch_data):
        logging.error(f"Length mismatch between sections and pitch data for {raag_dir}")
        return None

    return list(zip(sections, pitch_data))


def load_and_preprocess_data(root_path, max_raags=2):
    """Loads and preprocesses raag data, limiting the number of raags loaded."""
    print(f"Loading data from: {root_path}")
    all_output = []
    raag_count = 0

    for artist_dir in tqdm(os.listdir(root_path), desc="Processing Artists"):
        artist_path = os.path.join(root_path, artist_dir)
        if os.path.isdir(artist_path):
            for raag_dir in os.listdir(artist_path):
                if raag_count >= max_raags:
                    logging.info(f"Loaded data for {max_raags} raags. Stopping.")
                    break

                raag_path = os.path.join(artist_path, raag_dir)
                if os.path.isdir(raag_path):
                    # Find data files
                    sa_files = [f for f in os.listdir(raag_path) if f.endswith(".ctonic.txt")]
                    pitch_files = [f for f in os.listdir(raag_path) if f.endswith(".pitch.txt")]
                    sections_files = [f for f in os.listdir(raag_path) if f.endswith(".sections-manual-p.txt")]

                    if not sa_files or not pitch_files or not sections_files:
                        logging.warning(f"Skipping raag {raag_dir} due to missing files.")
                        continue

                    logging.info(f"Preprocessing raag: {raag_dir}")
                    output = preprocess_raag(raag_path, sa_files[0], pitch_files[0], sections_files[0])

                    if output:
                        all_output.extend(output)  # Extend all_output with preprocessed raag data
                        raag_count += 1

    logging.info(f"Total raags processed: {raag_count}")
    print(f"Total raags processed: {raag_count}")  # Print total raags processed
    return all_output  # Return the accumulated preprocessed data


def extract_all_notes(all_output):
    """Extracts all notes from the preprocessed data."""
    logging.info("Extracting all notes...")
    all_notes = []
    for artist_data in all_output:
        for song_data in artist_data['songs']:
            all_notes.extend(song_data['notes'])
    logging.info(f"Extracted {len(all_notes)} notes")
    return all_notes


def create_tokenizer(all_notes):
    """Creates a Keras tokenizer for the notes."""
    logging.info("Creating tokenizer...")
    tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=False, oov_token="<unk>")  # Use TensorFlow's tokenizer
    tokenizer.fit_on_texts(all_notes)
    logging.info("Tokenizer created.")
    return tokenizer


def create_sequences(tokenizer, all_notes, sequence_length, batch_size):
    """Transforms a list of notes into sequences suitable for training using tf.data.Dataset."""
    logging.info("Creating sequences...")

    # Convert all_notes to a TensorFlow tensor
    all_notes_tensor = tf.constant(all_notes)

    # Use tf.data.Dataset for sequence creation
    dataset = tf.data.Dataset.from_tensor_slices(all_notes_tensor)
    dataset = dataset.window(sequence_length + 1, shift=1, drop_remainder=True)  # Include target note
    dataset = dataset.flat_map(lambda window: window.batch(sequence_length + 1))
    dataset = dataset.map(lambda sequence: (
        [tokenizer.word_index.get(note.numpy().decode(), tokenizer.word_index['<unk>']) for note in sequence[:-1]],
        tokenizer.word_index.get(sequence[-1].numpy().decode(), tokenizer.word_index['<unk>'])
    ))

    # Batch the dataset
    dataset = dataset.batch(batch_size)

    # Prefetch for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset  # Return the tf.data.Dataset


def extract_raag_names(all_output):
    """Extracts the raag names from the dataset directory."""
    raag_names = set()  # Use a set for uniqueness
    for artist_data in all_output:
        for song_data in artist_data['songs']:
            raag_names.add(song_data['raag'])  # Add raag name to the set
    return list(raag_names)  # Convert the set back to a list


def create_raag_id_mapping(raag_names):
    """Creates a mapping from raag names to unique integer IDs."""
    raag_id_dict = {raag: i for i, raag in enumerate(raag_names)}
    return raag_id_dict, len(raag_id_dict)


def generate_raag_labels(root_path, all_notes, raag_id_dict, num_raags):
    """Generates raag labels for each sequence."""
    logging.info("Generating raag labels...")

    if not all_notes:
        logging.warning("No notes, cannot generate raag labels.")
        return np.zeros(len(all_notes), dtype='int32')  # Return zeros if no notes

    raag_labels = np.zeros(len(all_notes), dtype='int32')  # Initialize as 0s
    current_note_index = 0

    for raag_name, raag_id in raag_id_dict.items():
        raag_path = os.path.join(root_path, raag_name)

        # Find data files (assuming one file per type)
        sa_file = next((f for f in os.listdir(raag_path) if f.endswith(".ctonic.txt")), None)
        pitch_file = next((f for f in os.listdir(raag_path) if f.endswith(".pitch.txt")), None)
        sections_file = next((f for f in os.listdir(raag_path) if f.endswith(".sections-manual-p.txt")), None)

        if not all([sa_file, pitch_file, sections_file]):
            logging.warning(f"Skipping raag {raag_name} due to missing files.")
            continue

        preprocessed_data = preprocess_raag(raag_path, sa_file, pitch_file, sections_file)

        if preprocessed_data is None:
            logging.warning(f"Failed to load data for {raag_name}.")
            continue

        extracted_notes = [note for _, note in preprocessed_data if note is not None]

        # Optimized label assignment using list slicing and comparison
        num_matching_notes = len(extracted_notes)
        if current_note_index + num_matching_notes <= len(all_notes) and \
           all_notes[current_note_index: current_note_index + num_matching_notes] == extracted_notes:
            raag_labels[current_note_index: current_note_index + num_matching_notes] = raag_id
            current_note_index += num_matching_notes

    logging.info(f"Generated {len(raag_labels)} raag labels.")
    return raag_labels