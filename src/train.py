import os
import logging
import time
import yaml
import pickle
import tensorflow as tf
from models.data_utils import load_and_preprocess_data, extract_all_notes, create_sequences, create_raag_id_mapping, generate_raag_labels, tokenize_all_notes, create_tokenizer
from models.model_builder import create_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("train.log"), logging.StreamHandler()])


def main():
    logging.info("Starting train process...")
    start_time_total = time.time()

    # Check for GPU
    if tf.config.list_physical_devices('GPU'):
        logging.info("GPU is available and being used in train.py.")
        print("GPU is available and being used in train.py.")
    else:
        logging.warning("No GPU detected. Running on CPU in train.py.")
        print("No GPU detected. Running on CPU in train.py.")

    # Use the default strategy for CPU and single GPU
    strategy = tf.distribute.get_strategy()
    logging.info(f"REPLICAS in train.py: {strategy.num_replicas_in_sync}")

    if os.environ.get("COLAB_GPU", "FALSE") == "TRUE":
        repo_dir = "/content/drive/MyDrive/music_generation_repo"
        data_path = "/content/drive/MyDrive/"  # correct path
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
    epochs = config['epochs']
    batch_size = config['batch_size']
    validation_split = config['validation_split']
    patience = config['patience']
    model_name = config.get('model_name', 'my_model')
    tokenizer_name = config.get('tokenizer_name', 'my_tokenizer')

    # Data Preprocessing
    logging.info("Starting data preprocessing...")
    start_time = time.time()

    # Load and preprocess data once
    all_output, selected_raags, _ = load_and_preprocess_data(repo_dir, data_path)  # we dont need unique_raags
    if all_output is None or len(all_output) == 0:
        logging.error("No data was loaded. Check the data. Aborting")
        return

    # Extract all notes
    all_notes, all_output_filtered = extract_all_notes(all_output, selected_raags)

    if len(all_notes) == 0:
        logging.error("No notes extracted. Check data loading and preprocessing. Aborting.")
        return

    logging.info(f"Data preprocessing took {time.time() - start_time:.2f} seconds")

    # Tokenizer Creation
    logging.info("Creating Tokenizer...")
    start_time_tokenizer = time.time()
    tokenizer = create_tokenizer(all_notes)
    tokenizer_path = os.path.join(repo_dir, "tokenizers", f"{tokenizer_name}.pickle")
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    logging.info(f"Tokenizer saved to {tokenizer_path}")
    logging.info(f"Tokenizer creation took {time.time() - start_time_tokenizer:.2f} seconds")

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
    start_time_mapping = time.time()
    raag_id_dict, num_raags = create_raag_id_mapping(all_output_filtered, selected_raags)
    logging.info(f"Raag ID mapping complete. total: {num_raags}")
    logging.info(f"Raag ID mapping took {time.time() - start_time_mapping:.2f} seconds")

    if num_raags == 0:
        logging.error("No raags were found. Please check your data.")
        return

    # Generate raag labels
    logging.info("Generating raag labels...")
    start_time_labels = time.time()
    raag_labels = generate_raag_labels(all_output_filtered, raag_id_dict, num_raags, all_notes, sequence_length)
    logging.info("Raag labels generated")
    logging.info(f"Raag labels took {time.time() - start_time_labels:.2f} seconds")

    # Tokenize all notes
    start_time_tokenize = time.time()
    tokenized_notes = tokenize_all_notes(tokenizer, all_notes)
    logging.info(f"Tokenizing all notes took {time.time() - start_time_tokenize:.2f} seconds")

    logging.info(f"Sequence length: {sequence_length}")
    logging.info(f"Number of raag labels: {len(raag_labels)}")
    logging.info(f"Number of tokenized_notes: {len(tokenized_notes)}")

    # Create sequences
    sequences_dataset = create_sequences(tokenized_notes, sequence_length, batch_size * strategy.num_replicas_in_sync, raag_labels)

    # Model Creation and training
    with strategy.scope():
        model = create_model(vocab_size, num_raags, sequence_length, strategy)
        model_path = os.path.join(repo_dir, "models", f"{model_name}.h5")
        # Check if the model file exists before loading
        # Callbacks
        early_stopping = EarlyStopping(monitor='loss', patience=patience)
        model_checkpoint = ModelCheckpoint(filepath=model_path,
                                           monitor='loss',
                                           save_best_only=True,
                                           save_weights_only=True)

        # Train the model
        logging.info("Starting training...")
        start_time_training = time.time()
        model.fit(sequences_dataset, epochs=epochs, callbacks=[early_stopping, model_checkpoint])
        logging.info("Training finished.")
        logging.info(f"Training took {time.time() - start_time_training:.2f} seconds")

    logging.info(f"Total train execution time: {time.time() - start_time_total:.2f} seconds")
    logging.info("Train process completed.")

if __name__ == "__main__":
    main()

