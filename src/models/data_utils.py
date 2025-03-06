import os
import pretty_midi
import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict
import logging
import pickle
from tqdm import tqdm

def load_midi_file(midi_path: str) -> pretty_midi.PrettyMIDI:
    """
    Loads a MIDI file using pretty_midi and returns a PrettyMIDI object.

    Args:
        midi_path (str): The path to the MIDI file.

    Returns:
        pretty_midi.PrettyMIDI: The loaded MIDI file as a PrettyMIDI object.
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        return midi_data
    except Exception as e:
        logging.error(f"Failed to load MIDI file {midi_path}: {e}")
        return None

def midi_to_notes_list(midi_data: pretty_midi.PrettyMIDI) -> List[str]:
    """
    Extracts notes from a PrettyMIDI object and returns a list of note strings.

    Args:
        midi_data (pretty_midi.PrettyMIDI): The MIDI data.

    Returns:
        List[str]: A list of note strings.
    """
    notes_list = []
    try:
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                notes_list.append(f"{note.pitch},{note.start:.2f},{note.end:.2f},{note.velocity}")
    except Exception as e:
        logging.error(f"Failed to convert MIDI data to notes list: {e}")
        return []
    return notes_list

def process_raag_folder(raag_path: str) -> List[Dict]:
    """
    Processes a folder containing MIDI files for a specific raag.

    Args:
        raag_path (str): The path to the raag folder.

    Returns:
        List[Dict]: A list of dictionaries, each containing data for a MIDI file.
    """
    raag_name = os.path.basename(raag_path)
    raag_output = []

    for file_name in os.listdir(raag_path):
        if file_name.endswith(".mid"):
            midi_path = os.path.join(raag_path, file_name)
            midi_data = load_midi_file(midi_path)
            if midi_data is None:
                continue
            notes_list = midi_to_notes_list(midi_data)
            if not notes_list:
                continue
            raag_output.append({
                "raag": raag_name,
                "file_name": file_name,
                "notes": notes_list
            })
    return raag_output

def load_and_preprocess_data(repo_dir, data_path, num_raags_to_select=None) -> List[Dict]:
    """
    Loads and preprocesses data from raag folders.

    Args:
        repo_dir (str): The root directory of the repository.
        data_path (str): The path to the data directory.
        num_raags_to_select (int, optional): The number of raags to select. Defaults to None (all raags).

    Returns:
        List[Dict]: A list of dictionaries, each containing data for a MIDI file.
    """
    
    data_dir = os.path.join(data_path, "Dataset")
    all_output = []
    selected_raags = []
    if not os.path.exists(data_dir):
        logging.error(f"Data directory not found: {data_dir}")
        return []

    logging.info(f"Loading data from directory: {data_dir}")

    for raag_folder in os.listdir(data_dir):
        if num_raags_to_select is not None and len(selected_raags) >= num_raags_to_select:
            break

        raag_path = os.path.join(data_dir, raag_folder)
        if os.path.isdir(raag_path):
            if raag_folder not in selected_raags:
                selected_raags.append(raag_folder)
                logging.info(f"processing data from raag: {raag_folder}")
                all_output.extend(process_raag_folder(raag_path))
            else:
                logging.warning(f"Raag {raag_folder} already processed. Skipping.")
    return all_output

def extract_all_notes(data_output: List[Dict]) -> Tuple[List[str], List[Dict]]:
    """
    Extracts all notes from the data output and returns a list of notes and the filtered data.

    Args:
        data_output (List[Dict]): The data output.

    Returns:
        Tuple[List[str], List[Dict]]: A tuple containing the list of notes and the filtered data.
    """
    all_notes = []
    all_output_filtered = []
    for output_item in data_output:
        all_output_filtered.append(output_item)
        all_notes.extend(output_item["notes"])
    return all_notes, all_output_filtered

def create_tokenizer(notes_list: List[str]) -> tf.keras.preprocessing.text.Tokenizer:
    """
    Creates a tokenizer from a list of notes.

    Args:
        notes_list (List[str]): The list of notes.

    Returns:
        tf.keras.preprocessing.text.Tokenizer: The created tokenizer.
    """
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(notes_list)
    return tokenizer

def tokenize_all_notes(tokenizer: tf.keras.preprocessing.text.Tokenizer, all_notes: List[str]) -> List[int]:
    """
    Tokenizes a list of notes using the provided tokenizer.

    Args:
        tokenizer (tf.keras.preprocessing.text.Tokenizer): The tokenizer.
        all_notes (List[str]): The list of notes.

    Returns:
        List[int]: The list of tokenized notes.
    """
    return tokenizer.texts_to_sequences(all_notes)

def create_raag_id_mapping(all_output_filtered: List[Dict]) -> Tuple[Dict[str, int], int]:
    """
    Creates a raag ID mapping from the filtered data.

    Args:
        all_output_filtered (List[Dict]): The filtered data.

    Returns:
        Tuple[Dict[str, int], int]: A tuple containing the raag ID mapping and the number of raags.
    """
    raag_id_dict = {}
    num_raags = 0
    for item in all_output_filtered:
        raag_name = item["raag"]
        if raag_name not in raag_id_dict:
            raag_id_dict[raag_name] = num_raags
            num_raags += 1
    return raag_id_dict, num_raags

def generate_raag_labels(all_output_filtered: List[Dict], raag_id_dict: Dict[str, int], num_raags: int, all_notes: List[str], sequence_length: int) -> List[int]:
    """
    Generates raag labels for the notes based on the raag ID mapping.

    Args:
        all_output_filtered (List[Dict]): The filtered data.
        raag_id_dict (Dict[str, int]): The raag ID mapping.
        num_raags (int): The number of raags.
        all_notes (List[str]): The list of all notes.
        sequence_length (int): The sequence length.

    Returns:
        List[int]: The generated raag labels.
    """
    raag_labels = []
    for item in all_output_filtered:
        raag_name = item["raag"]
        raag_id = raag_id_dict[raag_name]
        # Generate sequence_length + 1 raag labels for each notes item
        num_notes = len(item["notes"])
        raag_labels.extend([raag_id] * num_notes)
    return raag_labels

def create_sequences(tokenized_notes: List[int], sequence_length: int, batch_size: int, raag_labels: List[int]) -> tf.data.Dataset:
    """
    Creates sequences from tokenized notes for training.

    Args:
        tokenized_notes (List[int]): The tokenized notes.
        sequence_length (int): The sequence length.
        batch_size (int): The batch size.
        raag_labels (List[int]): The raag labels.

    Returns:
        tf.data.Dataset: The created dataset.
    """
    tokenized_notes = np.array(tokenized_notes)
    raag_labels = np.array(raag_labels)

    # Combine notes and raag labels for sequence creation
    inputs = []
    outputs = []
    for i in range(0, len(tokenized_notes) - sequence_length, 1):
        seq_in = tokenized_notes[i:i + sequence_length]
        seq_out = tokenized_notes[i + sequence_length]
        raag_label = raag_labels[i + sequence_length]

        inputs.append((seq_in, raag_label))
        outputs.append(seq_out)

    # Convert to TensorFlow Dataset
    dataset = tf.data.Dataset.from_tensor_slices((np.array(inputs, dtype=object), np.array(outputs)))
    dataset = dataset.map(lambda x, y: ({"notes_input": x[0], "raag_label": x[1]}, y))
    
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

