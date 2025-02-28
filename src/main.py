import os
import logging
import datetime
import yaml
import pickle
import tensorflow as tf
from models.data_utils import load_tonic, load_pitch_data, load_sections, hz_to_svara, preprocess_raag, extract_all_notes, create_tokenizer, create_sequences, extract_raag_names, create_raag_id_mapping, generate_raag_labels, load_and_preprocess_data
from models.music_utils import generate_music, tokens_to_midi, generate_raag_music, generate_music_with_tonic, generate_random_seed, get_token_frequencies
from models.model_builder import create_model
from models.model import MultiHeadAttention, TransformerBlock, RaagConditioning
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    
    # Determine model save path
    model_save_name = input("Enter model name (or press Enter for default naming): ").strip()
    if not model_save_name:
        # Generate a name based on execution count (you would handle persistency)
        execution_count = 1
        model_save_name = f"model_{execution_count}"

    model_save_path = f"{model_save_name}.keras"

    # Determine tokenizer save path
    tokenizer_save_name = input("Enter tokenizer name (or press Enter for default naming): ").strip()
    if not tokenizer_save_name:
        # Generate a name based on execution count (you would handle persistency)
        execution_count = 1
        tokenizer_save_name = f"tokenizer_{execution_count}"
    
    tokenizer_save_path = f"{tokenizer_save_name}.pkl"
    
    # Data Preprocessing
    logging.info("Starting data preprocessing...")
    root_path = dataset_path
    all_output = load_and_preprocess_data(dataset_path)

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
    
    X, y = create_sequences(tokenizer, all_notes, sequence_length)
    if X is None or y is None:
        logging.error("Sequences not created successfully, check previous logs")
        return
    logging.info("Data preprocessing complete.")
    
    # Raag ID mapping
    logging.info("Creating raag ID mapping...")
    raag_id_dict, num_raags = create_raag_id_mapping(root_path)
    logging.info("Raag ID mapping complete")
    logging.info(f"Number of raags: {num_raags}")

    # Generate raag labels
    logging.info("Generating raag labels...")
    raag_labels = generate_raag_labels(root_path, X, all_notes, raag_id_dict, num_raags)
    logging.info("Raag labels generated")

    # Model Creation and Training
    logging.info("Starting model training...")
    model = create_model(vocab_size, num_raags, sequence_length)

    # Define the EarlyStopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience)

    # Define the ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        filepath='model_checkpoint_{epoch:02d}.h5',
        save_freq='epoch',
        monitor='val_loss',
        save_best_only=True
    )
    # Reshape raag_labels to (num_samples, 1)
    raag_labels = raag_labels.reshape(-1, 1)

    # Train the model with the callbacks
    history = model.fit(
        x=[X, raag_labels],
        y=y,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        callbacks=[early_stopping, checkpoint_callback]
    )
    logging.info("Model training complete.")
    # Save model and tokenizer
    try:
        model.save(model_save_path)
        logging.info(f"Model saved successfully to: {model_save_path}")
    except Exception as e:
        logging.error(f"An error occurred while saving the model: {e}")
    
    try:
        with open(tokenizer_save_path, 'wb') as f:
            pickle.dump(tokenizer, f)
        logging.info(f"Tokenizer saved successfully to: {tokenizer_save_path}")
    except Exception as e:
        logging.error(f"An error occurred while saving the tokenizer: {e}")

    # Music generation
    model = tf.keras.models.load_model(
        model_save_path,
        custom_objects={
            "RaagConditioning": RaagConditioning,
            "TransformerBlock": TransformerBlock,
            "MultiHeadAttention": MultiHeadAttention
        }
    )
    
    try:
        with open(tokenizer_save_path, 'rb') as f:
            tokenizer = pickle.load(f)
        logging.info("Tokenizer loaded successfully!")
    except Exception as e:
        logging.error(f"An error occurred while loading the tokenizer: {e}")

    # Music Generation with random seed
    logging.info("Generating Music with random seed...")
    seed_sequence = generate_random_seed(tokenizer, sequence_length)
    token_frequencies = get_token_frequencies(all_notes)
    raag_id_value = raag_id_dict.get('Basanti Kedar', 0)
    
    generated_tokens = generate_music(
        model, seed_sequence, raag_id_value, max_length=100, temperature=1.2, top_k=30, token_frequencies=token_frequencies
    )
    
    midi_data = tokens_to_midi(generated_tokens, tokenizer)
    midi_data.write(f"generated_music_raag_{raag_id_value}.mid")
    logging.info("Music generated and saved")

if __name__ == "__main__":
    main()