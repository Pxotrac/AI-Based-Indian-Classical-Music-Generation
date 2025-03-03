import os
import logging
import time
import datetime
import yaml
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from models.data_utils import load_and_preprocess_data, extract_all_notes, create_tokenizer, create_sequences, extract_raag_names, create_raag_id_mapping, generate_raag_labels
from models.music_utils import generate_music, tokens_to_midi, generate_raag_music, generate_music_with_tonic, generate_random_seed, get_token_frequencies
from models.model_builder import create_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
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
    # Load Config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    dataset_path = config['dataset_path'] 
    sequence_length = config['sequence_length']
    epochs = config['epochs']
    batch_size = config['batch_size']
    validation_split = config['validation_split']
    patience = config['patience']
    model_name = config.get('model_name', 'my_model')  # Get model_name from config, default to 'my_model'
    tokenizer_name = config.get('tokenizer_name', 'my_tokenizer')  # Get tokenizer_name, default to 'my_tokenizer'

    # Data Preprocessing
    logging.info("Starting data preprocessing...")
    start_time = time.time()  # Start timer

    # Load and preprocess data once
    all_output = load_and_preprocess_data(dataset_path)
    logging.info("Data loaded.")

    all_notes = extract_all_notes(all_output)
    if not all_notes:
        logging.warning("No notes were extracted during data preprocessing. Check data paths and formats.")
        return

    tokenizer = create_tokenizer(all_notes)
    if tokenizer is None:
        logging.error("Tokenizer was not created, check pre processing. Aborting")
        return

    vocab_size = len(tokenizer.word_index) + 1
    logging.info(f"Vocab size: {vocab_size}")

    if vocab_size <= 1:
        logging.error(f"The vocabulary size is too small: {vocab_size}. Please check your data.")
        return  # Stop execution if vocab size is too small

    # Create sequences using tf.data.Dataset
    sequences_dataset = create_sequences(tokenizer, all_notes, sequence_length, batch_size * strategy.num_replicas_in_sync)
    logging.info("Data preprocessing complete.")

    # Raag ID mapping
    logging.info("Creating raag ID mapping...")
    raag_id_dict, num_raags = create_raag_id_mapping(dataset_path)  # Assuming root_path is dataset_path
    logging.info("Raag ID mapping complete")
    logging.info(f"Number of raags: {num_raags}")

    # Generate raag labels
    logging.info("Generating raag labels...")
    raag_labels = generate_raag_labels(dataset_path, all_notes, raag_id_dict, num_raags)  # Removed X argument, use all_notes
    logging.info("Raag labels generated")

    end_time = time.time()  # End timer
    logging.info(f"Data preprocessing took {end_time - start_time:.2f} seconds")

    # Model Creation and Training within strategy.scope()
    with strategy.scope():
        model = create_model(vocab_size, num_raags, sequence_length, strategy)

        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        checkpoint_callback = ModelCheckpoint(filepath=f'{model_name}.h5', monitor='val_loss', save_best_weights=True)
   
    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  # Monitor validation loss
        patience=5,          # Number of epochs with no improvement before stopping
        restore_best_weights=True  # Restore the best model weights
    )
    # Train the model with early stopping and validation data
    history = model.fit(
        train_dataset,
        epochs=50,  # Adjust as needed
        validation_data=validation_dataset,  # Provide validation data
        callbacks=[early_stopping]
    )

    # Plot training history
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # Save the model and tokenizer
    model.save(f'{model_name}.h5')
    with open(f'{tokenizer_name}.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Music Generation with random seed
    logging.info("Generating Music with random seed...")
    seed_sequence = generate_random_seed(tokenizer, sequence_length)
    token_frequencies = get_token_frequencies(all_notes)

    # Get raag ID, handling potential KeyError
    raag_id_value = raag_id_dict.get('Basanti Kedar', 0)  
    if raag_id_value == 0 and 'Basanti Kedar' not in raag_id_dict:
        logging.warning("Raag 'Basanti Kedar' not found in raag ID dictionary. Using default ID 0.")

    generated_tokens = generate_music(
        model, seed_sequence, raag_id_value, max_length=100, temperature=1.2, top_k=30, token_frequencies=token_frequencies, strategy=strategy  # Pass strategy
    )

    midi_data = tokens_to_midi(generated_tokens, tokenizer)
    midi_data.write(f"generated_music_raag_{raag_id_value}.mid")
    logging.info("Music generated and saved")

if __name__ == "__main__":
    main()