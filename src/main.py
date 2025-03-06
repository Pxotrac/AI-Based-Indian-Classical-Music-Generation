import os
import logging
import time
import yaml
import pickle
import tensorflow as tf
from models.data_utils import load_and_preprocess_data, extract_all_notes, create_sequences, create_raag_id_mapping, generate_raag_labels, tokenize_all_notes
from models.music_utils import generate_music_with_tonic, generate_random_seed, get_token_frequencies, generate_raag_music, generate_music
from models.model_builder import create_model
from tqdm import tqdm
import subprocess
import sys
import random
import glob
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("main.log"), logging.StreamHandler()])

def run_script_with_live_output(script_path, repo_dir):
    """
    Runs a script using subprocess and prints its output in real-time.
    """
    logging.info(f"Running script from: {script_path}")
    try:
        os.chdir(os.path.join(repo_dir, "src"))

        process = subprocess.Popen(
            ["python", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env={**os.environ, "COLAB_GPU": "TRUE"}
        )

        os.chdir(repo_dir)

        while True:
            line = process.stdout.readline()
            if not line:
                break
            print(line, end='')
            logging.info(line.strip())

        process.wait()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, process.args)

        logging.info(f"Script '{script_path}' completed successfully.")

    except subprocess.CalledProcessError as e:
        logging.error(f"Error running script '{script_path}': {e}")
        return
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return

def sample_notes(all_notes, sample_percentage):
    num_to_sample = int(len(all_notes) * sample_percentage)
    sampled_notes = random.sample(all_notes, num_to_sample)
    return sampled_notes

def load_and_preprocess_data(repo_dir, data_path):
    """
    Loads and preprocesses MIDI data, randomly selects 35% of raags, and returns data from those raags.
    """
    logging.info("Loading and preprocessing data...")
    all_midi_files = glob.glob(os.path.join(data_path, "dataset/**/*.mid"), recursive=True)
    if len(all_midi_files) == 0:
        logging.error("No data was found. Check the data. Aborting")
        return None, None, None

    all_raags = [os.path.basename(os.path.dirname(f)) for f in all_midi_files]
    unique_raags = list(set(all_raags))
    logging.info(f"All unique raags found: {unique_raags}")

    num_raags_to_select = int(len(unique_raags) * 0.35)
    selected_raags = random.sample(unique_raags, num_raags_to_select)
    logging.info(f"Selected raags: {selected_raags}")

    filtered_midi_files = [f for f in all_midi_files if os.path.basename(os.path.dirname(f)) in selected_raags]

    all_output = []
    for midi_file in tqdm(filtered_midi_files, desc="Processing MIDI files"):
        output_file = os.path.join(repo_dir, "outputs", os.path.basename(midi_file)[:-4] + ".pickle")
        raag = os.path.basename(os.path.dirname(midi_file))
        all_output.append((raag, output_file))

    logging.info(f"Total raags Processed: {len(all_output)}")
    return all_output, selected_raags, unique_raags

def extract_all_notes(all_output, selected_raags):
    """
    Extracts all notes from the preprocessed data, filtering by selected raags.
    """
    all_notes = []
    all_notes_filtered = []
    for raag, file in all_output:
        with open(file, "rb") as f:
            notes = pickle.load(f)
            all_notes.extend(notes)
            if raag in selected_raags:
                all_notes_filtered.extend([(raag, notes)])
    logging.info(f"Total Notes: {len(all_notes)}")
    return all_notes, all_notes_filtered

def main():
    logging.info("Starting main process...")
    start_time = time.time()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate music in a specific raag.")
    parser.add_argument("--raag", type=str, help="The name of the raag to generate music in.")
    args = parser.parse_args()

    # Determine paths based on environment
    if os.environ.get("COLAB_GPU", "FALSE") == "TRUE":
        repo_dir = "/content/drive/MyDrive/music_generation_repo"
        data_path = "/content/drive/MyDrive/"
        print(f"Running on Colab. repo_dir: {repo_dir}")
        print(f"Running on Colab. data_path: {data_path}")
    else:
        repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.dirname(repo_dir)
        print(f"Running locally. repo_dir: {repo_dir}")
        print(f"Running locally. data_path: {data_path}")

    # Load Config
    config_file = os.path.join(repo_dir, "config.yaml")
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    sequence_length = config['sequence_length']
    model_name = config.get('model_name', 'MusicTransformer')
    tokenizer_name = config.get('tokenizer_name', 'transformer_tokenizer')
    batch_size = config['batch_size']
    sample_percentage = 0.35

    # GPU Check
    if tf.config.list_physical_devices('GPU'):
        logging.info("GPU is available and being used.")
        print("GPU is available and being used.")
    else:
        logging.warning("No GPU detected. Running on CPU.")
        print("No GPU detected. Running on CPU.")

    strategy = tf.distribute.get_strategy()
    logging.info(f"REPLICAS: {strategy.num_replicas_in_sync}")

    # Run train.py
    train_script_path = os.path.join("train.py")
    logging.info(f"Running train.py from: {train_script_path}")
    run_script_with_live_output(train_script_path, repo_dir)

    # Data Preprocessing
    logging.info("Starting data preprocessing...")
    start_time_preprocess = time.time()

    all_output, selected_raags, all_unique_raags = load_and_preprocess_data(repo_dir, data_path)
    if all_output is None or len(all_output) == 0:
        logging.error("No data was loaded. Check the data. Aborting")
        return

    # Save the available raags to a file
    available_raags_path = os.path.join(repo_dir, "available_raags.txt")
    with open(available_raags_path, "w") as f:
        for raag in all_unique_raags:
            f.write(f"{raag}\n")
    logging.info(f"List of all available raags saved to: {available_raags_path}")

    all_notes, all_output_filtered = extract_all_notes(all_output, selected_raags)

    if len(all_notes) == 0:
        logging.error("No notes extracted. Check data loading and preprocessing. Aborting.")
        return
    all_notes = sample_notes(all_notes, sample_percentage)

    # Load Tokenizer
    tokenizer_path = os.path.join(repo_dir, "tokenizers", f"{tokenizer_name}.pickle")
    if not os.path.exists(tokenizer_path):
        logging.warning(f"Tokenizer file not found at {tokenizer_path}. Please run train.py first.")
        return
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    if tokenizer is None:
        logging.error("Tokenizer was not created. Check preprocessing. Aborting")
        return
    if len(tokenizer.word_index) < 2:
        logging.error("Tokenizer created a vocabulary with less than 2 words. Check your notes data. Aborting.")
        return

    vocab_size = len(tokenizer.word_index) + 1
    logging.info(f"Vocab size: {vocab_size}")
    if vocab_size <= 1:
        logging.error(f"The vocabulary size is too small: {vocab_size}. Please check your data.")
        return

    # Raag ID mapping
    logging.info("Creating raag ID mapping...")
    raag_id_dict, num_raags = create_raag_id_mapping(all_output_filtered, selected_raags)
    logging.info(f"Raag ID mapping complete. total: {num_raags}")

    if num_raags == 0:
        logging.error("No raags were found. Please check your data.")
        return

    # Save the raag_id_dict to a file
    raag_id_dict_path = os.path.join(repo_dir, "raag_id_dict.pickle")
    with open(raag_id_dict_path, "wb") as f:
        pickle.dump(raag_id_dict, f)
    logging.info(f"Raag ID dictionary saved to: {raag_id_dict_path}")
    
    # Generate raag labels
    logging.info("Generating raag labels...")
    raag_labels = generate_raag_labels(all_output_filtered, raag_id_dict, num_raags, all_notes, sequence_length)
    logging.info("Raag labels generated")

    tokenized_notes = tokenize_all_notes(tokenizer, all_notes)
    logging.info(f"Sequence length: {sequence_length}")
    logging.info(f"Number of raag labels: {len(raag_labels)}")
    logging.info(f"Number of tokenized_notes: {len(tokenized_notes)}")

    sequences_dataset = create_sequences(tokenized_notes, sequence_length, batch_size * strategy.num_replicas_in_sync, raag_labels)
    logging.info("Data preprocessing complete.")

    end_time_preprocess = time.time()
    logging.info(f"Data preprocessing took {end_time_preprocess - start_time_preprocess:.2f} seconds")

    # Model Creation and generation
    with strategy.scope():
        model = create_model(vocab_size, num_raags, sequence_length, strategy)
        model_path = os.path.join(repo_dir, "models", f"{model_name}.h5")
        if not os.path.exists(model_path):
            logging.warning(f"Model file not found at {model_path}. Please run train.py first.")
            return

        model.load_weights(model_path)
        logging.info("Model loaded")

        logging.info("Generating Music with random seed...")
        seed_sequence = generate_random_seed(tokenizer, sequence_length)
        token_frequencies = get_token_frequencies(all_notes)

        # Get the raag ID
        if args.raag:
            raag_name = args.raag
        else:
            logging.info("No raag name provided. Selecting a random raag.")
            raag_name = random.choice(selected_raags)
        
        if raag_name not in raag_id_dict:
            logging.error(f"Raag '{raag_name}' is not in the list of selected raags. Available raags are {selected_raags}.")
            return
        
        raag_id_value = raag_id_dict[raag_name]
        logging.info(f"Generating music for raag: {raag_name} (ID: {raag_id_value})")

        generated_sequence = generate_music_with_tonic(model, seed_sequence, raag_id_value, tokenizer, max_length=100, temperature=1.2, top_k=30, strategy=strategy, vocab_size=vocab_size, sequence_length=sequence_length)
        generated_tokens_raag = generate_raag_music(model, raag_id_value, seed_sequence, tokenizer, max_length=100, temperature=1.2, top_k=30, strategy=strategy, vocab_size=vocab_size, sequence_length=sequence_length)

        logging.info("Music generated and saved")

        with open('generated_sequence.pickle', 'wb') as f:
            pickle.dump(generated_sequence, f)

    logging.info(f"Total execution time: {time.time() - start_time:.2f} seconds")
    logging.info("Main process completed.")

if __name__ == "__main__":
    main()
