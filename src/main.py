import os
import logging
import time
import yaml
import pickle
import tensorflow as tf
from models.data_utils import load_and_preprocess_data, extract_all_notes, create_tokenizer, create_raag_id_mapping, generate_raag_labels, tokenize_all_notes
from models.music_utils import generate_music_with_tonic, generate_random_seed, get_token_frequencies, generate_raag_music
from models.model_builder import create_model

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Use the default strategy for CPU and single GPU
strategy = tf.distribute.get_strategy()
print("REPLICAS: ", strategy.num_replicas_in_sync)

# --- Data Filtering ---
def filter_raags(all_output, raag_name=None):
    """
    Filters the dataset to include only the specified raag, or all raags if None.

    Args:
        all_output (list): The entire dataset.
        raag_name (str, optional): The name of the raag to select. Defaults to None (all raags).

    Returns:
        list, list: filtered dataset, list of selected raags.
    """
    if raag_name is None:
        logging.info("Generating music on the entire dataset.")
        return all_output, [item.get("raag") for item in all_output]  # Return all data and all raag names

    logging.info(f"Generating music only on raag: {raag_name}")

    filtered_data = [item for item in all_output if item.get("raag") == raag_name]
    return filtered_data, [raag_name]  # Return the filtered data and the selected raag name


def main(selected_model_name="indianraga_model", selected_raag=None):
    """
    Main function to generate music using a pre-trained model.

    Args:
        selected_model_name (str): The name of the model file (without extension) to use for generation.
        selected_raag(str, optional): The name of the raag to generate. If None, it will generate music in all raags.
    """
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
    repo_dir = os.path.abspath(repo_dir) #added this line, to avoid issues with relative paths.
    # Load Config - Now always load config.yaml with absolute path
    config_file = os.path.join(repo_dir, "config.yaml")
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    sequence_length = config['sequence_length']
    
    # Load Tokenizer
    tokenizer_path = os.path.join(repo_dir, "models", "ragatokenizer.pkl")
    if not os.path.exists(tokenizer_path):
        logging.error(f"Tokenizer file not found at {tokenizer_path}. Aborting.")
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
        
    # Model Creation and generation within strategy.scope()
    with strategy.scope():
        model = create_model(vocab_size, 10, sequence_length, strategy) #added 10 as num_raags, to load the model.
        # load the model
        model_path = os.path.join(repo_dir, "models", f"{selected_model_name}.keras")
        # Check if the model file exists before loading
        if not os.path.exists(model_path):
            logging.warning(f"Model file not found at {model_path}.")
            return

        model = tf.keras.models.load_model(model_path)
        logging.info("Model loaded")

        # Data Preprocessing
        logging.info("Starting data preprocessing for music generation...")

        # Load and preprocess data once
        all_output = load_and_preprocess_data(repo_dir, data_path)
        logging.info("Data loaded.")
        if all_output is None or len(all_output) == 0:
            logging.error("No data was loaded. Check the data. Aborting")
            return

        # Filter data by raag
        filtered_output, selected_raags = filter_raags(all_output, selected_raag)
        if selected_raags:
            logging.info(f"Selected raags for music generation: {selected_raags}")

        # Extract all notes
        all_notes, all_output_filtered = extract_all_notes(filtered_output)
        if not all_notes:
            logging.error("No notes extracted. Check data loading and preprocessing. Aborting.")
            return

        # Raag ID mapping
        logging.info("Creating raag ID mapping...")
        raag_id_dict, num_raags = create_raag_id_mapping(all_output_filtered)
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
        raag_labels = generate_raag_labels(all_output_filtered, raag_id_dict, num_raags, all_notes, sequence_length)
        logging.info("Raag labels generated")
        
        # Music Generation with random seed
        logging.info("Generating Music with random seed...")
        seed_sequence = generate_random_seed(tokenizer, sequence_length)
        #token_frequencies = get_token_frequencies(all_notes) # this is not required, as we are not using it.

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
    #Example calls:
    #main() #generate all raags with the default model.
    #main(selected_model_name="indianraga_model") #generate all raags with "indianraga_model.keras".
    main(selected_model_name="indianraga_model", selected_raag="Raag Bahar") #generate only Raag Bahar with "indianraga_model.keras".
