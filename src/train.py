import os
import logging
import time
import yaml
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from models.data_utils import load_and_preprocess_data, extract_all_notes, create_tokenizer, create_sequences, create_raag_id_mapping, generate_raag_labels, tokenize_all_notes
from models.model_builder import create_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Use the default strategy for CPU and single GPU
strategy = tf.distribute.get_strategy()
print("REPLICAS: ", strategy.num_replicas_in_sync)

def main():
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
    all_output = load_and_preprocess_data(repo_dir, data_path)
    logging.info("Data loaded.")
    if all_output is None or len(all_output) == 0:
        logging.error("No data was loaded. Check the data. Aborting")
        return

    # Extract all notes
    all_notes, all_output_filtered = extract_all_notes(all_output)
    if len(all_notes) == 0:
        logging.error("No notes extracted. Check data loading and preprocessing. Aborting.")
        return

    # Tokenization and vocabulary creation
    tokenizer = create_tokenizer(all_notes)
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
    raag_id_dict, num_raags = create_raag_id_mapping(all_output_filtered)
    logging.info("Raag ID mapping complete")
    logging.info(f"Number of raags: {num_raags}")
    if num_raags == 0:
        logging.error("No raags were found. Please check your data.")
        return

    # Generate raag labels
    logging.info("Generating raag labels...")
    raag_labels = generate_raag_labels(all_output_filtered, raag_id_dict, num_raags, all_notes, sequence_length)
    logging.info("Raag labels generated")

    # Tokenize all notes
    tokenized_notes = tokenize_all_notes(tokenizer, all_notes)
    logging.debug(f"Number of tokenized_notes: {len(tokenized_notes)}")
    if tokenized_notes:
        if len(tokenized_notes)>=10:
            logging.debug(f"First 10 tokenized_notes: {tokenized_notes[:10]}")
        
        logging.debug(f"First tokenized note: {tokenized_notes[0]}")
    else:
        logging.warning("tokenized_notes is empty.")

    # Check sequence length and raag labels length
    logging.info(f"Sequence length: {sequence_length}")
    logging.info(f"Number of raag labels: {len(raag_labels)}")
    logging.info(f"Batch size: {batch_size * strategy.num_replicas_in_sync}")

    logging.info(f"Dataset size before creation: {len(tokenized_notes) - sequence_length}")
    # Create sequences using tf.data.Dataset
    sequences_dataset = create_sequences(tokenized_notes, sequence_length, batch_size * strategy.num_replicas_in_sync, raag_labels)
    logging.info("Data preprocessing complete.")

    end_time = time.time()
    logging.info(f"Data preprocessing took {end_time - start_time:.2f} seconds")

    # Split the dataset in train and validation
    logging.info(f"Dataset size: {tf.data.experimental.cardinality(sequences_dataset).numpy()}")
    dataset_size = tf.data.experimental.cardinality(sequences_dataset).numpy()
    train_size = int(dataset_size * (1 - validation_split))
    train_dataset = sequences_dataset.take(train_size)
    validation_dataset = sequences_dataset.skip(train_size)

    # Model Creation, compilation and training within strategy.scope()
    with strategy.scope():
        model = create_model(vocab_size, num_raags, sequence_length, strategy)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

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
            verbose=1
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
        os.makedirs(os.path.join(repo_dir, "plots"), exist_ok=True)
        plt.savefig(os.path.join(repo_dir, "plots", f"{model_name}_training_history.png"))
        logging.info("Training history plot saved.")

if __name__ == "__main__":
    main()
