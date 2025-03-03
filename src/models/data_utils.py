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

def load_tonic(filepath):
    """Loads the tonic (Sa) frequency from the .ctonic.txt file."""
    tonic_file = os.path.splitext(filepath)[0] + ".ctonic.txt"
    try:
        with open(tonic_file, 'r') as f:
            tonic_hz_str = f.readline().strip()  # Read the first line and remove whitespace
            tonic_hz = float(tonic_hz_str)  # Convert to float
            return tonic_hz
    except FileNotFoundError:
        logging.warning(f"Tonic file not found: {tonic_file}")
        return None
    except ValueError:
        logging.warning(f"Invalid tonic value in file: {tonic_file}")
        return None
    except Exception as e:
        logging.error(f"Error loading file {tonic_file}: {e}")
        return None

def load_pitch_data(filepath):
    """Extracts pitch information from the .pitch.txt file."""
    pitch_file = os.path.splitext(filepath)[0] + ".pitch.txt"
    try:
        with open(pitch_file, 'r') as f:
            pitches = [float(line.strip()) for line in f]  # Assuming one pitch per line
            return pitches
    except FileNotFoundError:
        logging.warning(f"Pitch file not found: {pitch_file}")
        return None
    except ValueError:
        logging.warning(f"Invalid pitch value in file: {pitch_file}")
        return None
    except Exception as e:
        logging.error(f"Error loading file {pitch_file}: {e}")
        return None

def load_sections(filepath):
    """Loads section markers from the .sections-manual-p.txt file."""
    sections_file = os.path.splitext(filepath)[0] + ".sections-manual-p.txt"
    try:
        with open(sections_file, 'r') as f:
            sections_str = f.readline().strip()  # Read the first line
            sections = sections_str.split("|")  # Split by pipe symbol
            return sections
    except FileNotFoundError:
        logging.warning(f"Sections file not found: {sections_file}")
        return None
    except Exception as e:
        logging.error(f"Error loading file {sections_file}: {e}")
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

def load_and_preprocess_data(root_path, max_raags=None):
    print(f"Loading data from: {root_path}")
    all_output = []  # Initialize the list to store all processed data
    raag_count = 0  # Initialize a counter for the number of raags processed

    # Iterate over all subdirectories in the root_path
    for subdir, _, files in os.walk(root_path):
        # Check if the subdirectory is one level below the root directory
        if os.path.normpath(subdir).count(os.sep) - os.path.normpath(root_path).count(os.sep) >= 2:
            for file in files:
                # Check if the file is a .mp3 file
                if file.endswith(".mp3") and not file.endswith(".mp3.md5"):
                    try:
                        filepath = os.path.join(subdir, file)  # Full path to the file
                        raag_name = os.path.basename(os.path.dirname(filepath))  # Raag name is the subdirectory name

                        # Load data based on the current 'filepath'
                        tonic_hz = load_tonic(filepath)
                        pitch_data = load_pitch_data(filepath)
                        sections = load_sections(filepath)  # Corrected line
                        if pitch_data:
                            pitch_data = [hz_to_svara(pitch, tonic_hz) for pitch in pitch_data]  # Convert pitches to svaras
                        processed_data = {'raag': raag_name, 'tonic': tonic_hz, 'notes': pitch_data, 'sections': sections}  # Create data dictionary
                        all_output.append(processed_data)  # Add the processed data to the output list
                        raag_count += 1  # Increment the raag counter
                    except Exception as e:
                        logging.error(f"Error processing file {filepath}: {e}")  # Log an error if a file fails
    logging.info(f"Total raags processed: {raag_count}")  # Log the total number of raags processed
    print(f"Total raags processed: {raag_count}")  # Print total raags processed
    return all_output

def extract_all_notes(all_output):
    """Extracts all notes from the preprocessed data."""
    logging.info("Extracting all notes...")
    all_notes = []
    for data_point in all_output:
        notes = data_point.get('notes')
        if notes is not None:
            all_notes.append(notes)
        else:
            logging.warning("notes not found in data_point")
    logging.info(f"Extracted {len(all_notes)} notes")
    return all_notes


def create_tokenizer(all_notes):
    """Creates a Keras tokenizer for the notes."""
    logging.info("Creating tokenizer...")
    tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=False, oov_token="<unk>")  # Use TensorFlow's tokenizer
    tokenizer.fit_on_texts(all_notes)
    logging.info("Tokenizer created.")
    return tokenizer


def create_sequences(tokenizer, all_notes, sequence_length, batch_size, raag_labels):
    """Transforms a list of notes into sequences suitable for training using tf.data.Dataset."""
    logging.info("Creating sequences...")

    # Convert all_notes to a TensorFlow tensor
    all_notes_tensor = tf.constant(all_notes, dtype=tf.string)

    # Use tf.data.Dataset for sequence creation
    dataset = tf.data.Dataset.from_tensor_slices(all_notes_tensor)
    dataset = dataset.window(sequence_length + 1, shift=1, drop_remainder=True)  # Include target note
    dataset = dataset.flat_map(lambda window: window.batch(sequence_length + 1))
    dataset = dataset.map(lambda sequence: tokenize_sequence_tf(tokenizer, sequence))

    # Convert raag_labels to a TensorFlow tensor and create a dataset for it
    raag_labels_tensor = tf.constant(raag_labels, dtype=tf.int32)
    raag_labels_dataset = tf.data.Dataset.from_tensor_slices(raag_labels_tensor)
    # Create a combined dataset with sequences and corresponding raag labels
    dataset = tf.data.Dataset.zip((dataset, raag_labels_dataset))

    # Batch the combined dataset
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

def create_raag_id_mapping(all_output):
        """Creates a mapping from raag names to unique integer IDs."""
        raag_names = list({d['raag'] for d in all_output})  # Get unique raag names
        raag_id_dict = {raag: i for i, raag in enumerate(raag_names)}
        return raag_id_dict, len(raag_id_dict)

def generate_raag_labels(all_output, raag_id_dict, num_raags):
        """Generates raag labels for each data point in all_output."""
        logging.info("Generating raag labels")
        all_raag_labels = []  # To collect raag labels for all notes

        for data_point in all_output:
            raag_name = data_point['raag']  # Corrected: Access 'raag' directly
            
            # Correctly look up the raag_id
            raag_id = raag_id_dict.get(raag_name)
            if raag_id is None:
                logging.warning(f"Raag '{raag_name}' not found in raag ID dictionary. Skipping.")
                continue  # Skip this data point if the raag_id is not found
            if not data_point.get("notes"):
                continue
            
            notes_count = len(data_point['notes'])  # Get the number of notes in the data point
            raag_labels = [raag_id] * notes_count  # Create a list of raag_id repeated for each note
            all_raag_labels.extend(raag_labels)

        logging.info("Raag labels generated")
        return np.array(all_raag_labels, dtype='int32')
    
def tokenize_sequence(tokenizer, sequence):
    notes = sequence[:-1]
    target = sequence[-1]

    tokenized_notes = [tokenizer.word_index.get(note.decode(), tokenizer.word_index['<unk>']) for note in notes.numpy().tolist()]
    tokenized_target = tokenizer.word_index.get(target.decode(), tokenizer.word_index['<unk>'])
    return tokenized_notes, tokenized_target

def tokenize_sequence_tf(tokenizer, sequence):
    tokenized_notes, tokenized_target = tf.py_function(lambda x: tokenize_sequence(tokenizer, x), [sequence], [tf.int32, tf.int32])
    return tokenized_notes, tokenized_target
