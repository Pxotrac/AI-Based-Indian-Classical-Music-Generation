import os
import logging
import time
import yaml
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from models.data_utils import load_and_preprocess_data, extract_all_notes, create_tokenizer, create_sequences, create_raag_id_mapping, generate_raag_labels, tokenize_all_notes
from models.music_utils import generate_music_with_tonic, generate_random_seed, get_token_frequencies, generate_raag_music, generate_music
from models.model_builder import create_model
from tqdm import tqdm  # Import tqdm for progress bars
import subprocess
import sys
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("main.log"), logging.StreamHandler()])

# Use the default strategy for CPU and single GPU
strategy = tf.distribute.get_strategy()
print("REPLICAS: ", strategy.num_replicas_in_sync)

def run_script_with_live_output(script_path, repo_dir):
    """
    Runs a script using subprocess and prints its output in real-time.

    Args:
        script_path (str): The path to the script.
        repo_dir (str): The repository root directory.
    """
    logging.info(f"Running script from: {script_path}")
    try:
        os.chdir(os.path.join(repo_dir, "src")) #move to src folder before running

        process = subprocess.Popen(
            ["python", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            bufsize=1,  # Line buffering
            universal_newlines=True,
            env={**os.environ, "COLAB_GPU": "TRUE"}
        )
        
        os.chdir(repo_dir) #move back to repo_dir

        while True:
            line = process.stdout.readline()
            if not line:
                break
            print(line, end='')  # Print output to console
            logging.info(line.strip())  # Also log the output

        process.wait()  # Wait for process to finish
        
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, process.args)
        
        logging.info(f"Script '{script_path}' completed successfully.")

    except subprocess.CalledProcessError as e:
        logging.error(f"Error running script '{script_path}': {e}")
        return
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return

def main():
    logging.info("Starting main process...")
    start_time = time.time()

    # Determine paths based on the environment (Colab or local)
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

    # Load Config - Now always load config.yaml with absolute path
    config_file = os.path.join(repo_dir, "config.yaml")
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    sequence_length = config['sequence_length']
    model_name = config.get('model_name', 'MusicTransformer')
    tokenizer_name = config.get('tokenizer_name', 'transformer_tokenizer')
    batch_size = config['batch_size']

    # Run train.py using subprocess with live output
    train_script_path = os.path.join("train.py") #we dont need the full path anymore.
    logging.info(f"Running train.py from: {train_script_path}")
    run_script_with_live_output(train_script_path, repo_dir)


    # Data Preprocessing
    logging.info("Starting data preprocessing...")
    start_time = time.time()

    # Load and preprocess data once
    all_output = load_and_preprocess_data(repo_dir, data_path)
    logging.info("Data loaded.")
    if all_output is None or len(all_output) == 0:
        logging.error("No data was loaded. Check the data. Aborting")
        return

    # Extract all notes
    all_notes, all_output_filtered = extract_all_notes(all_output) #added filtered output
    
    if len(all_notes) == 0:
        logging.error("No notes extracted. Check data loading and preprocessing. Aborting.")
        return

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
    print(f"Tokenizer vocabulary: {tokenizer.word_index}")
    if vocab_size <= 1:
        logging.error(f"The vocabulary size is too small: {vocab_size}. Please check your data.")
        return

    # Raag ID mapping
    logging.info("Creating raag ID mapping...")
    raag_id_dict, num_raags = create_raag_id_mapping(all_output_filtered) #added filtered raag
    logging.info("Raag ID mapping complete")
    logging.info(f"Number of raags: {num_raags}")
    print(f"Raag ID dict: {raag_id_dict}")

    if num_raags == 0:
        logging.error("No raags were found. Please check your data.")
        return

    # Generate raag labels
    logging.info("Generating raag labels...")
    raag_labels = generate_raag_labels(all_output_filtered, raag_id_dict, num_raags, all_notes, sequence_length) #added filtered raag
    logging.info("Raag labels generated")

    # Tokenize all notes
    tokenized_notes = tokenize_all_notes(tokenizer, all_notes)
    logging.debug(f"Tokenized notes: {tokenized_notes[:5]}")  # Log the first 5 tokenized notes for debugging
    logging.debug(f"Raag labels: {raag_labels[:5]}")  # Log the first 5 raag labels for debugging
    logging.info(f"Sequence length: {sequence_length}")
    logging.info(f"Number of raag labels: {len(raag_labels)}")
    logging.info(f"Number of tokenized_notes: {len(tokenized_notes)}")
    if len(tokenized_notes)>10: logging.info(f"First 10 tokenized_notes: {tokenized_notes[:10]}")

    # Create sequences using tf.data.Dataset
    sequences_dataset = create_sequences(tokenized_notes, sequence_length, batch_size * strategy.num_replicas_in_sync, raag_labels)
    logging.info("Data preprocessing complete.")

    end_time = time.time()
    logging.info(f"Data preprocessing took {end_time - start_time:.2f} seconds")

    # Model Creation and generation within strategy.scope()
    with strategy.scope():
        model = create_model(vocab_size, num_raags, sequence_length, strategy)
        # load the model
        model_path = os.path.join(repo_dir, "models", f"{model_name}.h5")
        # Check if the model file exists before loading
        if not os.path.exists(model_path):
            logging.warning(f"Model file not found at {model_path}. Please run train.py first.")
            return

        model.load_weights(model_path)
        logging.info("Model loaded")
        # Music Generation with random seed
        logging.info("Generating Music with random seed...")
        seed_sequence = generate_random_seed(tokenizer, sequence_length)
        token_frequencies = get_token_frequencies(all_notes) # modified
        

        # Get raag ID, handling potential KeyError
        raag_name = 'Basanti Kedar'
        raag_id_value = raag_id_dict.get(raag_name, 0)
        if raag_id_value == 0 and raag_name not in raag_id_dict:
            logging.warning(f"Raag '{raag_name}' not found in raag ID dictionary. Using default ID 0.")
        # Music generation
        generated_sequence = generate_music_with_tonic(model, seed_sequence, raag_id_value, tokenizer, max_length=100, temperature=1.2, top_k=30, strategy=strategy, vocab_size=vocab_size, sequence_length=sequence_length)
        # Generate Raag music.
        generated_tokens_raag = generate_raag_music(model, raag_id_value, seed_sequence, tokenizer, max_length=100, temperature=1.2, top_k=30, strategy=strategy, vocab_size=vocab_size, sequence_length=sequence_length)

        logging.info("Music generated and saved")

        # Save the generated sequence
        with open('generated_sequence.pickle', 'wb') as f:
            pickle.dump(generated_sequence, f)
        
    logging.info(f"Total execution time: {time.time() - start_time:.2f} seconds")
    logging.info("Main process completed.")

if __name__ == "__main__":
    main()
