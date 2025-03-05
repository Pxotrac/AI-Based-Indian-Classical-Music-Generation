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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# TPU Initialization
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()  # Default strategy for CPU and single GPU

print("REPLICAS: ", strategy.num_replicas_in_sync)

def main():
    if os.environ.get("COLAB_GPU", "FALSE") == "TRUE":
        repo_dir = "/content/drive/MyDrive/music_generation_repo"
    else:
        repo_dir = os.path.dirname(os.path.abspath(__file__))
        repo_dir = os.path.dirname(repo_dir)  # Go up one more level

    # Load Config - Now always load config.yaml with absolute path
    config_file = os.path.join(repo_dir, "config.yaml")  # Use absolute path

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Data paths
    if os.environ.get("COLAB_GPU", "FALSE") == "TRUE":
        data_path = "/content/drive/MyDrive/"
        repo_dir = "/content/drive/MyDrive/music_generation_repo"
        print(f"Running on Colab. repo_dir: {repo_dir}")
        print(f"Running on Colab. data_path: {data_path}")
    else:
        repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.dirname(repo_dir)
        print(f"Running locally. repo_dir: {repo_dir}")
        print(f"Running locally. data_path: {data_path}")
    
    sequence_length = config['sequence_length']
    model_name = config.get('model_name', 'MusicTransformer')
    tokenizer_name = config.get('tokenizer_name', 'transformer_tokenizer')
    batch_size = config['batch_size'] *2 # Increased batch size

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

def check_expected_lengths(tokenized_notes, raag_labels, sequence_length):
    """Checks and logs the expected lengths of tokenized_notes, raag_labels, and sequence_length."""
    logging.info("Checking expected lengths...")
    logging.info(f"Expected length of raag_labels: {len(tokenized_notes) - sequence_length if tokenized_notes and sequence_length else 'N/A'}")
    logging.info(f"Actual length of tokenized_notes: {len(tokenized_notes)}")
    logging.info(f"Actual length of raag_labels: {len(raag_labels)}")
    logging.info(f"Sequence length: {sequence_length}")
    logging.info("Length check completed.")
if __name__ == "__main__":
    main()
