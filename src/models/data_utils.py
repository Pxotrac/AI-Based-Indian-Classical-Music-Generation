import os
import logging
import numpy as np
import pretty_midi
import re
from collections import Counter
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_tonic(file_path):
    """Loads tonic (Sa) frequency from a text file."""
    try:
        with open(file_path, 'r') as file:
            tonic = float(file.readline().strip())
        return tonic
    except Exception as e:
        logging.error(f"Error loading tonic from {file_path}: {e}")
        return None

def load_pitch_data(file_path, tonic_hz):
    """Loads pitch data from a text file, converting Hz to svara."""
    try:
        with open(file_path, 'r') as file:
            lines = [line.strip() for line in file]
            
        
        pitch_data = []
        for line in lines:
            try:
                pitch = float(line)
                svara = hz_to_svara(pitch, tonic_hz)
                pitch_data.append(svara)
            except ValueError:
                logging.warning(f"Skipping invalid line: {line}")
                continue

        return pitch_data
    except Exception as e:
        logging.error(f"Error loading pitch data from {file_path}: {e}")
        return None

def load_sections(file_path):
        """Loads sections from a text file."""
        try:
            with open(file_path, 'r') as file:
                sections = [line.strip() for line in file]
            return sections
        except Exception as e:
            logging.error(f"Error loading sections from {file_path}: {e}")
            return None

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
    
    pitch_data = load_pitch_data(pitch_path, tonic_hz)
    if pitch_data is None:
      logging.error(f"Failed to load pitch data for {raag_dir}")
      return None
    
    sections = load_sections(sections_path)
    if sections is None:
      logging.error(f"Failed to load sections for {raag_dir}")
      return None
      
    if len(sections) != len(pitch_data):
         logging.error(f"Length mismatch between sections and pitch data for {raag_dir}")
         return None

    return list(zip(sections, pitch_data))

def extract_all_notes(all_output):
    """Extracts all notes from the preprocessed data."""
    logging.info("Extracting all notes...")
    all_notes = []
    if not all_output:
        logging.warning("all_output is empty, no notes to extract")
        return []
    for raag_data in all_output:
        if raag_data is None:
            logging.warning("Skipping None raag_data")
            continue
        for section, note in raag_data:
            if note is not None:
                logging.info(f"appending note: {note} in section: {section}")
                all_notes.append(note)
            else:
                logging.warning(f"Skipping None note in section {section}")
    logging.info(f"Extracted {len(all_notes)} notes")
    return all_notes

def create_tokenizer(all_notes):
    """Creates a tokenizer from the list of all notes."""
    if not all_notes:
        logging.error("No notes to create tokenizer")
        return None
    
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token="<unk>")
    tokenizer.fit_on_texts(all_notes)
    logging.info("Tokenizer created")
    return tokenizer

def create_sequences(tokenizer, all_notes, sequence_length):
    """Transforms a list of notes into sequences suitable for training."""
    logging.info("Creating sequences...")
    sequences = []
    try:
        for i in range(len(all_notes) - sequence_length):
            seq = all_notes[i:i + sequence_length]
            seq_tokens = [tokenizer.word_index[note] for note in seq]
            sequences.append(seq_tokens)

        X = np.array(sequences[:-1])
        y = np.array([sequences[-1] for sequences in sequences[1:]])
        logging.info(f"Created {len(X)} sequences")
    except Exception as e:
        logging.error(f"Error creating sequences: {e}")
        return None, None

    logging.info("Sequences creation complete")
    return X, y

def extract_raag_names(root_path):
    """Extracts the raag names from the dataset directory."""
    raag_names = []
    for item in os.listdir(root_path):
        item_path = os.path.join(root_path, item)
        if os.path.isdir(item_path):
           raag_names.append(item)
    return raag_names

def create_raag_id_mapping(root_path):
    """Creates a mapping from raag names to unique integer IDs."""
    raag_names = extract_raag_names(root_path)
    raag_id_dict = {raag: i for i, raag in enumerate(raag_names)}
    return raag_id_dict, len(raag_id_dict)

def generate_raag_labels(root_path, X, all_notes, raag_id_dict, num_raags):
    """Generates raag labels for each sequence."""
    logging.info("Generating raag labels...")
    raag_labels = np.zeros(len(X), dtype='int32') # Initialize as 0s
    if not all_notes:
        logging.warning("No notes, cannot generate raag labels")
        return raag_labels
    
    current_note_index = 0
    
    for raag_name, raag_id in raag_id_dict.items():
      
      raag_path = os.path.join(root_path, raag_name)
      
      sa_file = [f for f in os.listdir(raag_path) if f.endswith(".sa.txt")]
      pitch_file = [f for f in os.listdir(raag_path) if f.endswith(".pitch.txt")]
      sections_file = [f for f in os.listdir(raag_path) if f.endswith(".sections-manual-p.txt")]
      
      if len(sa_file) == 0 or len(pitch_file) == 0 or len(sections_file) == 0:
          logging.warning(f"skipping raag {raag_name}")
          continue
      
      preprocessed_data = preprocess_raag(raag_path, sa_file[0], pitch_file[0], sections_file[0])

      if preprocessed_data is None:
          logging.warning(f"Failed to load data for {raag_name}")
          continue

      extracted_notes = [note for _, note in preprocessed_data if note is not None]
    
      for i in range(len(extracted_notes)):
        if current_note_index < len(all_notes) and all_notes[current_note_index] == extracted_notes[i]:
           raag_labels[current_note_index] = raag_id
           current_note_index += 1
        if current_note_index >= len(all_notes):
          break;
    logging.info("Raag labels generated")
    return raag_labels


def load_and_preprocess_data(dataset_path):
    """Loads and preprocesses all raag data."""
    logging.info("Starting data loading and preprocessing...")
    all_output = []
    
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path):
            sa_file = [f for f in os.listdir(item_path) if f.endswith(".sa.txt")]
            pitch_file = [f for f in os.listdir(item_path) if f.endswith(".pitch.txt")]
            sections_file = [f for f in os.listdir(item_path) if f.endswith(".sections-manual-p.txt")]
            
            if len(sa_file) == 0 or len(pitch_file) == 0 or len(sections_file) == 0:
              logging.error(f"Missing required files in {item_path}")
              continue
            
            preprocessed = preprocess_raag(item_path, sa_file[0], pitch_file[0], sections_file[0])
            if preprocessed is not None:
              all_output.append(preprocessed)
            else:
              logging.error(f"Error preprocessing {item_path}")
    logging.info(f"Finished loading and preprocessing, {sum([len(x) for x in all_output if x is not None])} total entries.")
    return all_output