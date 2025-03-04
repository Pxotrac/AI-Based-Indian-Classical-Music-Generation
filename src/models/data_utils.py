import os
import logging
import numpy as np
import pretty_midi
from collections import Counter
import tensorflow as tf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_tonic(filepath):
    """Loads the tonic (Sa) frequency from the .ctonic.txt file."""
    tonic_file = os.path.splitext(filepath)[0] + ".ctonic.txt"
    try:
        with open(tonic_file, 'r') as f:
            tonic_hz_str = f.readline().strip()
            tonic_hz = float(tonic_hz_str)
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
            pitches = [float(line.strip()) for line in f]
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
            sections_str = f.readline().strip()
            sections = sections_str.split("|")
            return sections
    except FileNotFoundError:
        logging.warning(f"Sections file not found: {sections_file}")
        return None
    except Exception as e:
        logging.error(f"Error loading file {sections_file}: {e}")
        return None
        
def hz_to_svara(frequency_hz, tonic_hz):
    """Converts frequency in Hz to a svara string with octave."""
    if not frequency_hz or frequency_hz == 0 or not tonic_hz or tonic_hz == 0:
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
    
    # Calculate the base octave and the number of octaves above the tonic
    octave = int(round(np.log2(frequency_hz / tonic_hz)))
    
    # Calculate the frequency in the same octave as the tonic
    adjusted_frequency = frequency_hz / (2 ** octave)
    
    closest_svara = None
    min_diff = float('inf')
    
    for svara, ratio in ratios.items():
        svara_freq = tonic_hz * ratio
        diff = abs(adjusted_frequency - svara_freq)
        if diff < min_diff:
            min_diff = diff
            closest_svara = svara
            
    if closest_svara is not None:
        return f"{closest_svara}{octave}"
    else:
        return None

def load_and_preprocess_data(root_path, data_path, max_raags=None): #change
    """Loads and preprocesses data from the dataset directory."""
    print(f"Loading data from: {data_path}") #change
    logging.info(f"Loading data from: {data_path}") #change
    all_output = []
    raag_count = 0
    
    # Check if the 'hindustani' folder exists
    if os.path.exists(os.path.join(data_path, "hindustani")): #change
        dataset_folder = os.path.join(data_path, "hindustani") #change
        logging.info(f"Dataset folder found: {dataset_folder}")
    else:
        logging.error(f"Dataset not found in path: {os.path.join(data_path, 'hindustani')}") #change
        return []

    for artist_folder in os.listdir(dataset_folder):
        artist_path = os.path.join(dataset_folder, artist_folder)
        logging.info(f"Processing artist folder: {artist_path}")  # New logging
        if os.path.isdir(artist_path):
            for raag_folder in os.listdir(artist_path):
                raag_path = os.path.join(artist_path, raag_folder)
                logging.info(f"  Processing raag folder: {raag_path}")  # New logging
                if os.path.isdir(raag_path):
                    for file in os.listdir(raag_path):
                        if file.endswith(".pitch.txt"):
                            try:
                                filepath = os.path.join(raag_path, file)
                                logging.info(f"    Processing file: {filepath}")  # New logging
                                raag_name = os.path.basename(raag_path)
                                
                                # load the data with the pitch filepath
                                tonic_hz = load_tonic(filepath)
                                pitch_data = load_pitch_data(filepath)
                                sections = load_sections(filepath)
                                if pitch_data:
                                    pitch_data = [hz_to_svara(pitch, tonic_hz) for pitch in pitch_data]
                                processed_data = {'raag': raag_name, 'tonic': tonic_hz, 'notes': pitch_data, 'sections': sections}
                                all_output.append(processed_data)
                                raag_count += 1
                            except Exception as e:
                                logging.error(f"Error processing file {filepath}: {e}")
    if raag_count == 0:
        logging.error("No raags found. Please check your dataset structure.")

    logging.info(f"Total raags processed: {raag_count}")
    print(f"Total raags processed: {raag_count}")
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
    tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=False, oov_token="<unk>")
    tokenizer.fit_on_texts(all_notes)
    logging.info("Tokenizer created.")
    return tokenizer

def tokenize_all_notes(tokenizer, all_notes):
    """Tokenizes a list of notes using the provided tokenizer."""
    tokenized_notes = []
    for note_list in all_notes:
        tokens = [tokenizer.word_index.get(note, tokenizer.word_index['<unk>']) for note in note_list]
        tokenized_notes.extend(tokens)
    return tokenized_notes

def create_sequences(tokenized_notes, sequence_length, batch_size, raag_labels):
    """Transforms a list of tokenized notes into sequences suitable for training using tf.data.Dataset."""
    logging.info("Creating sequences...")
    all_notes_tensor = tf.constant(tokenized_notes, dtype=tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices(all_notes_tensor)
    dataset = dataset.window(sequence_length + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(sequence_length + 1))
    dataset = dataset.map(lambda sequence: split_into_features_and_target(sequence))

    raag_labels_tensor = tf.constant(raag_labels, dtype=tf.int32)
    raag_labels_dataset = tf.data.Dataset.from_tensor_slices(raag_labels_tensor)
    dataset = tf.data.Dataset.zip((dataset, raag_labels_dataset))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
def split_into_features_and_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text


def extract_raag_names(all_output):
    """Extracts the raag names from the dataset directory."""
    raag_names = set()  # Use a set for uniqueness
    for data_point in all_output:
      raag_names.add(data_point['raag'])
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
