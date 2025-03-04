import os
import logging
import time
import datetime
import yaml
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from models.data_utils import load_and_preprocess_data, extract_all_notes, create_tokenizer, create_sequences, create_raag_id_mapping, generate_raag_labels, tokenize_all_notes
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
    # Load Config - Now always load config.yaml
    config_file = "config.yaml"  # Always use config.yaml
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
if os.environ.get("COLAB_GPU", "FALSE") == "TRUE":
    repo_dir = "/content/drive/MyDrive/music_generation_repo"
    data_path = "/content/drive/MyDrive/"  #correct path
else:
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(repo_dir)  # Go up one more level
    data_path = os.path.dirname(os.path.abspath(__file__)) #
    data_path = os.path.dirname(data_path)
    data_path = os.path.dirname(data_path)

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
    all_output = load_and_preprocess_data(repo_dir, data_path) #change
    logging.info("Data loaded.")

    # Extract all notes
    all_notes = extract_all_notes(all_output)  # Function to extract all notes from all_output
    all_notes_flatten = [item for sublist in all_notes for item in sublist]
    if len(all_notes_flatten) == 0:
       logging.error("No notes extracted. Check data loading and preprocessing. Aborting.")
       return

    # Tokenization and vocabulary creation
    tokenizer = create_tokenizer(all_notes_flatten)
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
        return  # Stop execution if vocab size is too small
    
    # Raag ID mapping
    logging.info("Creating raag ID mapping...")
    raag_id_dict, num_raags = create_raag_id_mapping(all_output)  # Create mapping from processed data
    logging.info("Raag ID mapping complete")
    logging.info(f"Number of raags: {num_raags}")
    if num_raags == 0:
      logging.error("No raags were found. Please check your data.")
      return

    # Generate raag labels
    logging.info("Generating raag labels...")
    raag_labels = generate_raag_labels(all_output, raag_id_dict, num_raags)  # Generate labels from processed data
    logging.info("Raag labels generated")
    
    # Tokenize all notes
    tokenized_notes = tokenize_all_notes(tokenizer, all_notes)
    
    # Create sequences using tf.data.Dataset
    sequences_dataset = create_sequences(tokenized_notes, sequence_length, batch_size * strategy.num_replicas_in_sync, raag_labels)
    logging.info("Data preprocessing complete.")
    
    end_time = time.time()  # End timer
    logging.info(f"Data preprocessing took {end_time - start_time:.2f} seconds")

    # Split the dataset in train and validation
    dataset_size = tf.data.experimental.cardinality(sequences_dataset).numpy()
    train_size = int(dataset_size * (1 - validation_split))
    train_dataset = sequences_dataset.take(train_size)
    validation_dataset = sequences_dataset.skip(train_size)

    # Model Creation, compilation and training within strategy.scope()
    with strategy.scope():
        model = create_model(vocab_size, num_raags, sequence_length, strategy)

         # Model Checkpoint
        checkpoint_filepath = os.path.join(repo_dir, "checkpoints", f"{model_name}.h5")
        model_checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True)

        # Early Stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

        # Training
        logging.info("Starting model training...")
        training_start_time = time.time()
        history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=epochs,
            callbacks=[early_stopping, model_checkpoint_callback],
            verbose=1  # Set verbose=1 to see progress bars
        )
        training_end_time = time.time()
        logging.info(f"Model training took {training_end_time - training_start_time:.2f} seconds")

        # Save the model
        model_path = os.path.join(repo_dir, "models", f"{model_name}.h5")
        model.save(model_path)
        logging.info(f"Model saved to {model_path}")

        # Save tokenizer
        tokenizer_path = os.path.join(repo_dir, "tokenizers", f"{tokenizer_name}.pickle")
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(tokenizer, f)
        logging.info(f"Tokenizer saved to {tokenizer_path}")

        # Plot training history
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.title('Training and Validation Loss')

        plt.tight_layout()
        os.makedirs(os.path.join(repo_dir,"plots"), exist_ok=True) # create the path if it doesn't exist
        plt.savefig(os.path.join(repo_dir,"plots", f"{model_name}_training_history.png"))
        logging.info("Training history plot saved.")

if __name__ == "__main__":
    main()
