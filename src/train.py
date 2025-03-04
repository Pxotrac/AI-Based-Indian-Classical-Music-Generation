import os
import logging
import time
import datetime
import yaml
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from models.data_utils import load_and_preprocess_data, extract_all_notes, create_tokenizer, create_sequences, extract_raag_names, create_raag_id_mapping, generate_raag_labels
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

    # Extract all notes
    all_notes = extract_all_notes(all_output)  # Function to extract all notes from all_output
    all_notes = [item for sublist in all_notes for item in sublist]

    # Tokenization and vocabulary creation
    tokenizer = create_tokenizer(all_notes)
    if tokenizer is None:
        logging.error("Tokenizer was not created. Check preprocessing. Aborting")
        return

    vocab_size = len(tokenizer.word_index) + 1
    logging.info(f"Vocab size: {vocab_size}")

    if vocab_size <= 1:
        logging.error(f"The vocabulary size is too small: {vocab_size}. Please check your data.")
        return  # Stop execution if vocab size is too small
    
     # Raag ID mapping
    logging.info("Creating raag ID mapping...")
    raag_id_dict, num_raags = create_raag_id_mapping(all_output)  # Create mapping from processed data
    logging.info("Raag ID mapping complete")
    logging.info(f"Number of raags: {num_raags}")

    # Generate raag labels
    logging.info("Generating raag labels...")
    raag_labels = generate_raag_labels(all_output, raag_id_dict, num_raags)  # Generate labels from processed data
    logging.info("Raag labels generated")
    
    # Create sequences using tf.data.Dataset
    sequences_dataset = create_sequences(tokenizer, all_notes, sequence_length, batch_size * strategy.num_replicas_in_sync, raag_labels)
    logging.info("Data preprocessing complete.")
    
    # Split dataset into training and validation sets
    # Calculate split indices
    dataset_size = tf.data.experimental.cardinality(sequences_dataset).numpy()  # Get dataset size
    validation_size = int(dataset_size * validation_split)  # Calculate validation size
    train_size = dataset_size - validation_size  # Calculate training size

    # Split the dataset
    validation_dataset = sequences_dataset.take(validation_size)  # Take validation part
    train_dataset = sequences_dataset.skip(validation_size)  # Skip validation and take rest

    # Log dataset sizes
    logging.info(f"Training dataset size: {train_size}")
    logging.info(f"Validation dataset size: {validation_size}")

    end_time = time.time()  # End timer
    logging.info(f"Data preprocessing took {end_time - start_time:.2f} seconds")

    # Model Creation and Training within strategy.scope()
    with strategy.scope():
        model = create_model(vocab_size, num_raags, sequence_length, strategy)

        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        checkpoint_callback = ModelCheckpoint(filepath=f'{model_name}.h5', monitor='val_loss', save_best_weights=True)
   
        # Train the model with early stopping and validation data
        history = model.fit(
            train_dataset,
            epochs=epochs,  # Adjust as needed
            validation_data=validation_dataset,  # Provide validation data
            callbacks=[early_stopping, checkpoint_callback]
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
   
if __name__ == "__main__":
    main()
