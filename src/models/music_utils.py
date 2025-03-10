import tensorflow as tf
import logging
import random
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_random_seed(tokenizer, sequence_length):
    """Generates a random seed sequence of the specified length."""
    logging.info("Generating random seed sequence...")
    vocab_size = len(tokenizer.word_index) + 1  # +1 to account for OOV token
    random_seed = [random.randint(1, vocab_size - 1) for _ in range(sequence_length)]
    logging.info(f"Generated seed sequence: {random_seed}")
    return random_seed

def get_token_frequencies(all_notes):
    """Calculates the frequency of each token in the entire dataset."""
    logging.info("Calculating token frequencies...")
    token_counts = {}
    for note in all_notes:
        if note not in token_counts:
            token_counts[note] = 0
        token_counts[note] += 1

    # Sort tokens by frequency in descending order
    sorted_tokens = sorted(token_counts.items(), key=lambda item: item[1], reverse=True)
    logging.info(f"Token frequencies: {sorted_tokens[:10]}")
    return sorted_tokens

def generate_music(model, seed_sequence, raag_id, tokenizer, max_length, temperature, top_k, strategy, vocab_size, sequence_length):
    """
    Generates music from a seed sequence.

    Args:
        model (tf.keras.Model): The trained music generation model.
        seed_sequence (list): The starting sequence (seed).
        raag_id (int): The raag ID for generation.
        tokenizer (tf.keras.preprocessing.text.Tokenizer): The tokenizer for converting between notes and tokens.
        max_length (int): The maximum length of the generated sequence.
        temperature (float): The temperature for sampling.
        top_k (int): The top-k value for sampling.
        strategy (tf.distribute.Strategy): The TensorFlow distribution strategy.
        vocab_size (int): the vocabuilary size.
        sequence_length (int): the sequence length

    Returns:
        list: The generated music sequence as a list of tokens.
    """
    generated_sequence = []
    current_sequence = seed_sequence.copy()
    logging.info(f"Generating music with seed: {seed_sequence} and raag_id {raag_id}...")
    for _ in range(max_length):
        input_sequence = np.array([current_sequence[-sequence_length:]])
        input_sequence = tf.constant(input_sequence, dtype=tf.int32)
        # Reshape raag_id to match the expected shape of the raag_label input
        raag_label = np.array(raag_id)  # Shape: ()
        raag_label = tf.constant(raag_label, dtype=tf.int32)
        
        predictions = model([input_sequence, raag_label], training=False)

        predictions = tf.squeeze(predictions, axis=0)

        predictions = predictions / temperature
        
        # Apply top-k filtering
        top_k_predictions, top_k_indices = tf.math.top_k(predictions, k=top_k)
        top_k_probabilities = tf.nn.softmax(top_k_predictions)

        # Sample from the filtered distribution
        sampled_index = tf.random.categorical([top_k_probabilities], num_samples=1, dtype=tf.int32)
        sampled_index = tf.squeeze(sampled_index, axis=[0, 1])

        predicted_token = tf.gather(top_k_indices, sampled_index)

        predicted_token_value = predicted_token.numpy().item()
        generated_sequence.append(predicted_token_value)
        current_sequence.append(predicted_token_value)

    logging.info(f"Generated music: {generated_sequence}")
    return generated_sequence

def generate_music_with_tonic(model, seed_sequence, raag_id, tokenizer, max_length, temperature, top_k, strategy, vocab_size, sequence_length):
    """Generates music with a given tonic and raag."""
    logging.info("Generating music with tonic...")
    return generate_music(model, seed_sequence, raag_id, tokenizer, max_length, temperature, top_k, strategy, vocab_size, sequence_length)

def generate_raag_music(model, raag_id, seed_sequence, tokenizer, max_length, temperature, top_k, strategy, vocab_size, sequence_length):
    """Generates music for a specific raag."""
    logging.info("Generating raag music...")
    return generate_music(model, seed_sequence, raag_id, tokenizer, max_length, temperature, top_k, strategy, vocab_size, sequence_length)
