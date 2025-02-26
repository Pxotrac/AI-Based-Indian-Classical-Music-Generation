import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
import logging
import re
from glob import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_tonic(ctonic_path):
    """Load tonic frequency from .ctonic.txt file (handles scalar/array)"""
    try:
        tonic_data = np.loadtxt(ctonic_path)
        return tonic_data.item() if tonic_data.ndim == 0 else tonic_data[0]
    except FileNotFoundError:
        logging.warning(f"Tonic file not found {ctonic_path}, using a default value")
        return 440
    except Exception as e:
        logging.error(f"Error loading tonic: {e}")
        return 440

def load_pitch_data(pitch_path):
    """Load time-stamped pitch values from .pitch.txt"""
    try:
        return np.loadtxt(pitch_path)
    except FileNotFoundError:
        logging.warning(f"Pitch file not found: {pitch_path}")
        return np.array([])
    except Exception as e:
        logging.error(f"Error loading pitch data: {e}")
        return np.array([])
def load_sections(sections_path):
    """Load sections from comma-separated .sections-manual-p.txt files"""
    sections = []
    if os.path.exists(sections_path):
        try:
            with open(sections_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(",")
                    if len(parts) < 4:
                        continue
                    start = float(parts[0])
                    end = float(parts[2])
                    # Remove non-ASCII characters and normalize
                    label = parts[3].encode("ascii", "ignore").decode().strip().lower()
                    label = "_".join(label.split())  # Replace spaces with underscores
                    sections.append((start, end, label))
        except FileNotFoundError:
            logging.warning(f"Sections file not found: {sections_path}")
        except Exception as e:
            logging.error(f"Error loading sections: {str(e)}")
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
        # Construct paths based on the files inside the folders, not the folder names
        
        ctonic_files = glob(os.path.join(raag_folder, "*.ctonic.txt"))
        pitch_files = glob(os.path.join(raag_folder, "*.pitch.txt"))
        sections_files = glob(os.path.join(raag_folder, "*.sections-manual-p.txt"))
        
        print(f"Checking folder: {raag_folder}") # Debugging
        
        if ctonic_files:
            ctonic_path = ctonic_files[0]
        else:
            raise FileNotFoundError(f"Missing .ctonic.txt file in {raag_folder}")

        if pitch_files:
            pitch_path = pitch_files[0]
        else:
            raise FileNotFoundError(f"Missing .pitch.txt file in {raag_folder}")

        if sections_files:
            sections_path = sections_files[0]
        else:
            raise FileNotFoundError(f"Missing .sections-manual-p.txt file in {raag_folder}")
            
        print(f"  ctonic_path: {ctonic_path}") # Debugging
        print(f"  pitch_path: {pitch_path}")   # Debugging
        print(f"  sections_path: {sections_path}") # Debugging

        # Load data
        tonic_hz = load_tonic(ctonic_path)
        pitch_data = load_pitch_data(pitch_path)
        sections = load_sections(sections_path)
               
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
                "section": label,
                "type": "section_start"
            })
        
        return note_sequence
    except Exception as e:
        logging.error(f"Error processing {raag_folder}: {str(e)}")
        return []


def load_and_preprocess_data(dataset_path):
    """Loads data from specified path and performs necessary preprocessing."""
    logging.info(f"Loading and preprocessing data from: {dataset_path}")
    output = []
    for root, dirs, files in os.walk(dataset_path):
        for dir in dirs:
            raag_folder = os.path.join(root, dir)
            output.extend(preprocess_raag(raag_folder))
    logging.info(f"Finished loading and preprocessing, {len(output)} total entries.")
    return output

def extract_all_notes(output):
    all_notes = []
    for entry in output:
        if "svara" in entry:
            all_notes.append(entry["svara"])
        elif "section" in entry:
            all_notes.append(entry["section"])
    return all_notes

def create_tokenizer(all_notes):
    """Creates and fits a Tokenizer, checks if all_notes is empty"""
    if not all_notes:
        logging.warning("No notes found, tokenizer will not be fitted.")
        return None  # Or raise an exception, depending on how you want to handle it
    tokenizer = Tokenizer(
        filters='',
        lower=False,
        oov_token="<UNK>",
        split=None,
        char_level=False
    )
    tokenizer.fit_on_texts(all_notes)
    return tokenizer

def create_sequences(tokenizer, all_notes, sequence_length):
    if not all_notes:
         logging.warning("No notes found, sequences cannot be created.")
         return np.array([]), np.array([])
    token_ids = [tokenizer.texts_to_sequences([note])[0][0] for note in all_notes]
    X, y = [], []
    for i in range(len(token_ids) - sequence_length):
        X.append(token_ids[i:i+sequence_length])
        y.append(token_ids[i+sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    return X, y

def extract_raag_names(folder_name):
    """Extract raag names from folder names."""
    import re
    folder_name = re.sub(r'\bby\b.*|\b&\b.*|\(.*?\)|\[.*?\]', '', folder_name, flags=re.IGNORECASE)
    raags = []
    for part in re.split(r',|&|/|_', folder_name):
        part = part.strip()
        if part and len(part) > 3:
            raags.append(part.title().strip())
    return list(set(raags))

def create_raag_id_mapping(root_path):
    all_raag_names = set()
    for root, dirs, files in os.walk(root_path):
        for dir_name in dirs:
            if "Raag" in dir_name:
                extracted_raags = extract_raag_names(dir_name)
                all_raag_names.update(extracted_raags)
    
    unique_raags = sorted(list(all_raag_names))
    raag_id_dict = {raag: idx for idx, raag in enumerate(unique_raags)}
    return raag_id_dict, len(all_raag_names)

def generate_raag_labels(root_path, X, all_notes, raag_id_dict, num_raags):
        import numpy as np
        logging.info("Generating raag labels...")
        raag_labels = []
        
        all_output = []
        for root, dirs, files in os.walk(root_path):
            for dir in dirs:
                if "Raag" in dir:
                    raag_folder = os.path.join(root, dir)
                    output = preprocess_raag(raag_folder)
                    if output:
                        all_output.append(output)

        output_index = 0
        for root, dirs, files in os.walk(root_path):
            for dir_name in dirs:
                if "Raag" in dir_name:
                  raag_names = extract_raag_names(dir_name)
                  if raag_names:
                      raag_id_value = raag_id_dict.get(raag_names[0], 0)
                      num_notes_for_raag = 0
                      
                      #Count notes for current raag
                      if output_index < len(all_output):
                          num_notes_for_raag = len(all_output[output_index])
                      
                      raag_labels.extend([raag_id_value] * num_notes_for_raag)
                      output_index +=1

        raag_labels = np.pad(raag_labels, (0, len(X) - len(raag_labels)), 'constant')
        logging.info("Raag labels generated successfully")
        return np.array(raag_labels)