import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense
from tensorflow.keras.models import Model
import keras_nlp
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_model(vocab_size, num_raags, sequence_length):
    """
    Creates and returns the transformer-based model for music generation.
    Args:
        vocab_size (int): Size of the vocabulary (number of unique tokens).
        num_raags (int): Number of unique raags in the dataset.
        sequence_length (int): Length of input sequences.
    Returns:
        tf.keras.Model: The compiled model.
    """
    logging.info("Creating model...")
    
    # Inputs
    note_input = Input(shape=(sequence_length,), dtype=tf.int32, name="note_input")  # Specify dtype
    raag_input = Input(shape=(1,), dtype=tf.int32, name="raag_input")
    
    note_embed = Embedding(vocab_size, 64)(note_input)
    raag_embed = Embedding(num_raags, 64)(raag_input)  # Now matches your 58 raags
    combined = note_embed + raag_embed
    
    # Transformer (using keras_nlp)
    transformer = keras_nlp.layers.TransformerEncoder(
        intermediate_dim=128,  # Adjust if needed
        num_heads=8,
        dropout=0.2
    )(combined)
    
    # Output
    output = Dense(vocab_size, activation="softmax")(transformer[:, -1, :])
    
    model = Model(inputs=[note_input, raag_input], outputs=output)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    
    logging.info("Model created successfully")
    return model
