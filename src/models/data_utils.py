import os
import logging
import numpy as np
import pretty_midi
from collections import Counter
import tensorflow as tf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_tonic(filepath):
    """Loads the tonic (Sa) frequency from the .ctonic.txt file."""
    directory = os.path.dirname(filepath) #added
    file_name = os.path.basename(filepath).replace(".pitch.txt",".ctonic.txt") # added
    tonic_file = os.path.join(directory,file_name) #added
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
                line = line.strip() #added
                if '\t' in line: #added
                    parts = line.split('\t') # added
                    line = parts[0]  #added take only the first part
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
    directory = os.path.dirname(filepath) #added
    file_name = os.path.basename(filepath).replace(".pitch.txt",".sections-manual-p.txt") # added
    sections_file = os.path.join(directory,file_name) #added
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

def load_and_preprocess_data(repo_dir, data_path, max_raags=None): #modified removed min notes
    """Loads and preprocesses data from the dataset directory."""
    print(f"Loading data from: {data_path}") #change
    logging.info(f"Loading data from: {data_path}") #change
    all_output = []
    raag_count = 0
    
    # Check if the 'hindustani' folder exists
    dataset_folder = os.path.join(data_path, "hindustani","hindustani") #change
    logging.info(f"Checking for dataset folder at: {dataset_folder}") #added
    if not os.path.exists(dataset_folder): # added
                logging.error(f"Dataset not found in path: {dataset_folder}. There is no 'hindustani' folder inside 'hindustani'") #changed
                return []
    else:
        logging.info(f"Dataset folder found: {dataset_folder}")

    for artist_folder in os.listdir(dataset_folder):
        artist_path = os.path.join(dataset_folder, artist_folder)
        logging.info(f"Processing artist folder: {artist_path}")  # New logging
        if os.path.isdir(artist_path):
            for raag_folder in os.listdir(artist_path):#added raag loop
                raag_path = os.path.join(artist_path, raag_folder) #added raag path
                logging.info(f"  Processing raag folder: {raag_path}")  # New logging
                if os.path.isdir(raag_path): #added verify raag is a directory
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
    if raag_count == 0:
        logging.error("No raags found. Please check your dataset structure.")

    logging.info(f"Total raags processed: {raag_count}")
    print(f"Total raags processed: {raag_count}")
    return all_output

def extract_all_notes(all_output, min_notes=100):
    """Extracts all notes from the preprocessed data."""
    logging.info("Extracting all notes...")
    all_notes = []
    all_output_filtered = [] #added array
    for data_point in all_output:
        notes = data_point.get('notes')
        if notes is None:
            logging.warning(f"No notes found for {data_point.get('raag')}. Skipping.")
            continue
        if len(notes) > min_notes: # added filter
            all_notes.extend(notes) #modified
            all_output_filtered.append(data_point) # added save data point
        else: #added if less than min_notes
            logging.warning(f"Skipping raag {data_point.get('raag')} because it has less than {min_notes} notes.")#added
            continue # added
    logging.info(f"Total number of notes found: {len(all_notes)}")
    return all_notes, all_output_filtered # changed

def create_tokenizer(all_notes):
    """Creates a tokenizer based on the unique notes."""
    if not all_notes or len(all_notes) == 0:
        logging.error("No notes provided to create a tokenizer. Aborting.")
        return None

    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', lower=False, oov_token="<unk>")  # Removed filters and set lower=False
    tokenizer.fit_on_texts(all_notes) # modified
    return tokenizer

def tokenize_all_notes(tokenizer, all_notes):
    """Tokenizes all notes using the provided tokenizer."""
    logging.info("Tokenizing all notes...")
    tokenized_notes = tokenizer.texts_to_sequences(all_notes)
    return tokenized_notes

def tokenize_sequence(tokenizer, sequence):
    """Tokenizes a sequence of notes using the provided tokenizer."""
    tokenized_sequence = tokenizer.texts_to_sequences([sequence])  # Pass as a list of strings
    return tokenized_sequence[0]  # Return the first (and only) list of tokenized sequences

def create_sequences(tokenized_notes, sequence_length, batch_size, raag_labels):
    """Creates sequences from tokenized notes, batching and adding raag labels."""
    logging.info("Creating sequences...")
    sequences_with_labels = []
    for i, seq in enumerate(tokenized_notes): #changed
        if len(seq) < sequence_length + 1:  # Add +1 to account for the target note
          continue
        for j in range(0, len(seq) - sequence_length):
            input_sequence = seq[j:j+sequence_length]
            # Check if raag_labels has enough labels
            if len(raag_labels) > i: # changed
              target_raag_id = raag_labels[j]  # Get raag_id from raag_labels
              sequences_with_labels.append((input_sequence, target_raag_id, seq[j+sequence_length]))
            else:
              logging.warning(f"Not enough labels for sequence {i}. Skipping.")

    # Shuffle and batch the data
    np.random.shuffle(sequences_with_labels)
    features, raag_ids, targets = zip(*sequences_with_labels) #modified
    features_padded = tf.keras.preprocessing.sequence.pad_sequences(features)  # Pad the sequences
    # Convert to tensors
    features_tensor = tf.convert_to_tensor(features_padded, dtype=tf.int32)
    targets_tensor = tf.convert_to_tensor(targets, dtype=tf.int32)
    raag_ids_tensor = tf.convert_to_tensor(raag_ids, dtype=tf.int32)
    
    dataset = tf.data.Dataset.from_tensor_slices(((features_tensor, raag_ids_tensor), targets_tensor)) #modified
    dataset = dataset.batch(batch_size) #modified

    logging.info(f"Total number of sequences created: {len(sequences_with_labels)}")
    return dataset

def split_into_features_and_target_raag(sequence, raag_id): #modified
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

def generate_raag_labels(all_output, raag_id_dict, num_raags): #added
    """Generates raag labels for each sequence based on the raag ID mapping."""
    logging.info("Generating raag labels...")
    all_raag_labels = []
    
    for data_point in all_output:
        raag_name = data_point['raag']  # Corrected: Access 'raag' directly
        
        # Correctly look up the raag_id
        raag_id = raag_id_dict.get(raag_name)
        if raag_id is None:
            logging.warning(f"Raag '{raag_name}' not found in raag ID dictionary. Skipping.")
            continue

        notes_count = len(data_point.get('notes')) # Get the number of notes in the data point
        raag_labels = [raag_id] * notes_count  # Create a list of raag_id repeated for each note
        all_raag_labels.extend(raag_labels)# modified

    logging.info(f"Total raag labels generated: {len(all_raag_labels)}")
    return all_raag_labels
