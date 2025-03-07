import os
import logging
import time
import yaml
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from models.data_utils import load_and_preprocess_data, extract_all_notes, create_tokenizer, create_raag_id_mapping, generate_raag_labels, tokenize_all_notes, tokenize_sequence
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
# --- Data Filtering ---
def filter_raags(all_output, num_raags_to_select=None):
    """
    Filters the dataset to include only the first 'num_raags_to_select' raags.

    Args:
        all_output (list): The entire dataset.
        num_raags_to_select (int, optional): The number of raags to select. Defaults to None (all raags).

    Returns:
        list, list: filtered dataset, list of selected raags.
    """
    if num_raags_to_select is None:
        logging.info("Generating music on the entire dataset.")
        return all_output, []  # Return all data and empty list of selected raags

    logging.info(f"Generating music on the first {num_raags_to_select} raags.")
    
    unique_raags = []
    filtered_data = []

    for item in all_output:
        raag_name = item.get("raag")
        if raag_name not in unique_raags:
          unique_raags.append(raag_name)
          if len(unique_raags) >= num_raags_to_select:
            break # end of raags
        
    selected_raags = unique_raags[:num_raags_to_select]
    # filter dataset
    for item in all_output:
        if item.get("raag") in selected_raags:
            filtered_data.append(item)
    
    return filtered_data, selected_raags

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
    #model_name = config.get('model_name', 'MusicTransformer') #removed as we are not using the model_name
    #tokenizer_name = config.get('tokenizer_name', 'transformer_tokenizer') #removed as we are not using the tokenizer_name
    num_raags_to_select = config.get("num_raags_to_select", None)

    # Run train.py using subprocess with live output (commented)
    #train_script_path = os.path.join("train.py") #commented, as we dont need to run it.
    #logging.info(f"Running train.py from: {train_script_path}") #commented, as we dont need to run it.
    #run_script_with_live_output(train_script_path, repo_dir) #commented, as we dont need to run it.


    # Data Preprocessing
    logging.info("Starting data preprocessing for music generation...")

    # Load and preprocess data once
    all_output = load_and_preprocess_data(repo_dir, data_path, num_raags_to_select) #added num_raags_to_select
    logging.info("Data loaded.")
    if all_output is None or len(all_output) == 0:
        logging.error("No data was loaded. Check the data. Aborting")
        return

    # Filter data by raag
    filtered_output, selected_raags = filter_raags(all_output, num_raags_to_select)
    if selected_raags:
        logging.info(f"Selected raags for music generation: {selected_raags}")

    # Extract all notes
    all_notes, all_output_filtered = extract_all_notes(filtered_output) #added filtered_output
    if not all_notes:
        logging.error("No notes extracted. Check data loading and preprocessing. Aborting.")
        return

    # Load Tokenizer
    #tokenizer_path = os.path.join(repo_dir, "tokenizers", f"{tokenizer_name}.pickle") #commented old code
    tokenizer_path = os.path.join(repo_dir,"models","ragatokenizer.pkl") #added, to use the trained tokenizer
    if not os.path.exists(tokenizer_path):
        logging.warning(f"Tokenizer file not found at {tokenizer_path}.")
        return
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    if not tokenizer:
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
    raag_id_dict, num_raags = create_raag_id_mapping(all_output_filtered) #added filtered_output
    logging.info("Raag ID mapping complete")
    logging.info(f"Number of raags: {num_raags}")
    print(f"Raag ID dict: {raag_id_dict}")

    if num_raags == 0:
        logging.error("No raags were found. Please check your data.")
        return
    # Check that there is no error in raag_id_dict
    if not raag_id_dict:
        logging.error("The dictionary raag_id_dict is empty. Aborting.")
        return

    # Generate raag labels
    logging.info("Generating raag labels...")
    raag_labels = generate_raag_labels(all_output_filtered, raag_id_dict, num_raags, all_notes, sequence_length) #added filtered_output
    logging.info("Raag labels generated")

    # Model Creation and generation within strategy.scope()
    with strategy.scope():
        model = create_model(vocab_size, num_raags, sequence_length, strategy)
        # load the model
        #model_path = os.path.join(repo_dir, "models", f"{model_name}.keras") #commented old code
        model_path = os.path.join(repo_dir,"models","indianraga_model.keras")#added correct path for our trained model.
        # Check if the model file exists before loading
        if not os.path.exists(model_path):
            logging.warning(f"Model file not found at {model_path}.")
            return

        #model.load_weights(model_path) #commented old code
        model = tf.keras.models.load_model(model_path) #added new code to load entire model
        logging.info("Model loaded")

        # Music Generation with random seed
        logging.info("Generating Music with random seed...")
        seed_sequence = generate_random_seed(tokenizer, sequence_length)
        token_frequencies = get_token_frequencies(all_notes)

        # Loop over the selected raags to generate music for each
        for raag_name in selected_raags:
            logging.info(f"Generating music for raag: {raag_name}")

            # Get raag ID, handling potential KeyError
            raag_id_value = raag_id_dict.get(raag_name)
            if raag_id_value is None:
                logging.warning(f"Raag '{raag_name}' not found in raag ID dictionary. Skipping this raag.")
                continue

            # Music generation
            generated_sequence = generate_music_with_tonic(model, seed_sequence, raag_id_value, tokenizer, max_length=100, temperature=1.2, top_k=30, strategy=strategy, vocab_size=vocab_size, sequence_length=sequence_length)
            # Generate Raag music.
            generated_tokens_raag = generate_raag_music(model, raag_id_value, seed_sequence, tokenizer, max_length=100, temperature=1.2, top_k=30, strategy=strategy, vocab_size=vocab_size, sequence_length=sequence_length)

            logging.info(f"Music generated for raag: {raag_name} and saved")

            # Save the generated sequence
            with open(f'generated_sequence_{raag_name}.pickle', 'wb') as f:
                pickle.dump(generated_sequence, f)

            # Save the generated Raag music
            with open(f'generated_tokens_raag_{raag_name}.pickle', 'wb') as f:
                pickle.dump(generated_tokens_raag, f)
    
    logging.info(f"Total execution time: {time.time() - start_time:.2f} seconds")
    logging.info("Main process completed.")

if __name__ == "__main__":
    main()
