import os
import logging
import numpy as np
import pretty_midi
from collections import Counter
import tensorflow as tf
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_tonic(filepath):
    """Loads the tonic (Sa) frequency from the .ctonic.txt file."""
    directory = os.path.dirname(filepath)
    file_name = os.path.basename(filepath).replace(".pitch.txt",".ctonic.txt")
    tonic_file = os.path.join(directory,file_name)
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
    pitches = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if '\t' in line:
                    parts = line.split('\t')
                    line = parts[0]
                try:
                    pitch = float(line)
                    pitches.append(pitch)
                except ValueError:
                    logging.warning(f"Invalid pitch value found in file: {filepath}. Skipping this line.")
                    continue
            return pitches
    except FileNotFoundError:
        logging.warning(f"Pitch file not found: {filepath}")
        return None
    except Exception as e:
        logging.error(f"Error loading file {filepath}: {e}")
        return None

def load_sections(filepath):
    """Loads section markers from the .sections-manual-p.txt file."""
    directory = os.path.dirname(filepath)
    file_name = os.path.basename(filepath).replace(".pitch.txt",".sections-manual-p.txt")
    sections_file = os.path.join(directory,file_name)
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

def load_and_preprocess_data(repo_dir, data_path, num_raags_to_select=None): #added num_raags_to_select
    """Loads and preprocesses data from the dataset directory."""
    print(f"Loading data from: {data_path}")
    logging.info(f"Loading data from: {data_path}")
    all_output = []
    raag_count = 0
    selected_raags = []
    
    # Check if the 'hindustani' folder exists
    dataset_folder = os.path.join(data_path, "hindustani","hindustani")
    logging.info(f"Checking for dataset folder at: {dataset_folder}")
    if not os.path.exists(dataset_folder):
        logging.error(f"Dataset not found in path: {dataset_folder}. There is no 'hindustani' folder inside 'hindustani'")
        return []
    else:
        logging.info(f"Dataset folder found: {dataset_folder}")
    
    for artist_folder in os.listdir(dataset_folder):
        artist_path = os.path.join(dataset_folder, artist_folder)
        logging.info(f"Processing artist folder: {artist_path}")  # New logging
        if os.path.isdir(artist_path):
            for raag_folder in os.listdir(artist_path):#added raag loop
                if num_raags_to_select is not None and len(selected_raags) >= num_raags_to_select:
                    logging.info(f"Reached maximum number of raags to select: {num_raags_to_select}")
                    break
                if raag_folder in selected_raags:
                    logging.info(f"Raag {raag_folder} already selected. Skipping.")
                    continue
                raag_path = os.path.join(artist_path, raag_folder) #added raag path
                logging.info(f"  Processing raag folder: {raag_path}")  # New logging
                if os.path.isdir(raag_path): #added verify raag is a directory
                    selected_raags.append(raag_folder)
                    for file in os.listdir(raag_path):
                        if file.endswith(".pitch.txt"):
                            try:
                                filepath = os.path.join(raag_path, file)
                                logging.info(f"    Processing file: {filepath}")  # New logging
                                raag_name = os.path.basename(raag_path)
                                
                                # load the data with the pitch filepath
                                if os.path.exists(filepath):# Verify that the file exist.
                                    tonic_hz = load_tonic(filepath)
                                    pitch_data = load_pitch_data(filepath)
                                    sections = load_sections(filepath)
                                    if pitch_data:
                                        pitch_data = [hz_to_svara(pitch, tonic_hz) for pitch in pitch_data if pitch !=0] # Only add not 0
                                        if len(pitch_data) > 0:
                                            processed_data = {'raag': raag_name, 'tonic': tonic_hz, 'notes': pitch_data, 'sections': sections}
                                            all_output.append(processed_data)
                                            raag_count += 1
                                else:
                                    logging.warning(f"Pitch file not found: {filepath}")
                            except Exception as e:
                                logging.error(f"Error processing file {filepath}: {e}")
                else:
                    logging.warning(f"Raag path {raag_path} is not a directory, skipping")

    if raag_count == 0:
        logging.error("No raags found. Please check your dataset structure.")

    logging.info(f"Total raags processed: {raag_count}")
    print(f"Total raags processed: {raag_count}")
    return all_output

def extract_all_notes(all_output, min_notes=100):
    """Extracts all notes from the preprocessed data."""
    logging.info("Extracting all notes...")
    all_notes = []
    all_output_filtered = []
    for data_point in all_output:
        if len(data_point.get('notes', [])) >= min_notes:
            all_notes.extend(data_point['notes'])  # Use .get('notes', []) to safely handle missing keys
            all_output_filtered.append(data_point)
        else:
            logging.warning(f"Skipping raag {data_point.get('raag')} because it has less than {min_notes} notes.")
            
    logging.info(f"Total number of notes found: {len(all_notes)}")
    logging.info(f"Number of filtered raags: {len(all_output_filtered)}")
    return all_notes, all_output_filtered

def create_tokenizer(all_notes):
    """Creates a tokenizer based on the unique notes."""
    if not all_notes or len(all_notes) == 0:
        logging.error("No notes provided to create a tokenizer. Aborting.")
        return None

    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', lower=False, oov_token="<unk>")
    tokenizer.fit_on_texts(all_notes)
    return tokenizer

def tokenize_all_notes(tokenizer, all_notes):
    """Tokenizes all notes using the provided tokenizer."""
    logging.info("Tokenizing all notes...")
    start_time = time.time()  # Start timer
    tokenized_notes = tokenizer.texts_to_sequences(all_notes)
    end_time = time.time()  # End timer
    logging.info(f"Tokenizing all notes took {end_time - start_time:.2f} seconds")
    return [item for sublist in tokenized_notes for item in sublist]

def tokenize_sequence(tokenizer, sequence):
    """Tokenizes a sequence of notes using the provided tokenizer."""
    tokenized_sequence = tokenizer.texts_to_sequences([sequence])  # Pass as a list of strings
    return tokenized_sequence[0]  # Return the first (and only) list of tokenized sequences

def create_sequences(tokenized_notes, sequence_length, batch_size, raag_labels):
    """Creates sequences and labels for model training using tf.data.Dataset."""
    logging.info("Creating sequences...")
    start_time = time.time()

    # Check if tokenized_notes is empty or if sequence_length is invalid
    if not tokenized_notes or sequence_length <= 0:
        logging.warning("No sequences created. Check your input data and parameters.")
        return tf.data.Dataset.from_tensor_slices(([], [])).batch(batch_size)  # Return an empty dataset
    # Check if sequence_length is greater than the length of tokenized_notes
    if sequence_length >= len(tokenized_notes):
        logging.warning("Sequence length is greater than or equal to the length of tokenized_notes.")
        return tf.data.Dataset.from_tensor_slices(([], [])).batch(batch_size)
    # Check if there are enough raag labels
    if len(raag_labels) < len(tokenized_notes) - sequence_length:
        logging.warning("Not enough raag labels to create sequences.")
        return tf.data.Dataset.from_tensor_slices(([], [])).batch(batch_size)  # Return an empty dataset
    # Check batch size
    if batch_size <= 0:
        logging.warning("Batch size is invalid")
        return tf.data.Dataset.from_tensor_slices(([], [])).batch(batch_size)
    logging.info("Creating dataset...")
    
    # Convert to TensorFlow tensors
    tokenized_notes_tensor = tf.constant(tokenized_notes, dtype=tf.int32)
    raag_labels_tensor = tf.constant(raag_labels, dtype=tf.int32)
    
    # Create sequences and next_notes
    sequences_dataset = tf.data.Dataset.from_tensor_slices(((tokenized_notes_tensor[:-sequence_length],raag_labels_tensor[sequence_length:]),tokenized_notes_tensor[sequence_length:]))
    sequences_dataset = sequences_dataset.batch(batch_size)
    end_time = time.time()
    logging.info(f"Dataset created in: {end_time - start_time:.2f} seconds")
    logging.info(f"Dataset elements: {tf.data.experimental.cardinality(sequences_dataset)}")

    return sequences_dataset

def split_into_features_and_target_raag(sequence, raag_id):
    """Splits a sequence into features and target, and returns the raag ID."""
    input_sequence = sequence[:-1]  # All but the last element
    target = sequence[-1]  # Last element is the target
    return input_sequence, target, raag_id  # Return the input sequence, target, and raag ID

def extract_raag_names(all_output):
    """Extracts unique raag names from the dataset."""
    raag_names = set()
    for data_point in all_output:
        raag_names.add(data_point['raag'])
    return list(raag_names)

def create_raag_id_mapping(all_output):
    """Creates a mapping from raag names to unique IDs."""
    logging.info("Creating raag ID mapping...")
    raag_names = extract_raag_names(all_output)
    raag_id_dict = {raag_name: i for i, raag_name in enumerate(raag_names)}
    logging.info(f"Raag ID mapping created: {raag_id_dict}")
    logging.info(f"Total unique raags found: {len(raag_id_dict)}")
    return raag_id_dict, len(raag_id_dict)

def generate_raag_labels(all_output_filtered, raag_id_dict, num_raags, all_notes, sequence_length):
    """Generates raag labels for the entire dataset."""
    logging.info("Generating raag labels...")
    start_time = time.time()  # Start timer
    raag_labels = []
    for entry in all_output_filtered:
        raag_id = raag_id_dict.get(entry['raag'], -1)  # Get raag ID from the dictionary, default to -1 if not found
        if raag_id != -1:
            raag_labels.extend([raag_id] * len(entry['notes']))  # Assign raag ID to all notes in the entry
        else:
             logging.warning(f"Raag not found in raag_id_dict: {entry['raag']}. This raag will be ignored")
            
    if len(raag_labels) < (len(all_notes) - sequence_length):
        logging.warning(f"The length of raag_labels ({len(raag_labels)}) is less than the length of tokenized_notes - sequence_length ({len(all_notes) - sequence_length}).")
    
    if not raag_labels:
        logging.error("No valid raag labels were generated. Please check your data.")

    end_time = time.time()  # End timer
    logging.info(f"Total raag labels generated: {len(raag_labels)}")
    logging.info(f"Raag labels generated in {end_time - start_time:.2f} seconds")
    return raag_labels
