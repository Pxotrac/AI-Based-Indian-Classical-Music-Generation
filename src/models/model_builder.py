import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense
from tensorflow.keras.models import Model
import keras_nlp

def create_model(vocab_size, num_raags, sequence_length):
    # Inputs
    note_input = Input(shape=(sequence_length,), dtype=tf.int32, name="note_input")  # Specify dtype
    raag_input = Input(shape=(1,), dtype=tf.int32, name="raag_input")

    note_embed = Embedding(vocab_size, 64)(note_input)
    raag_embed = Embedding(num_raags, 64)(raag_input)
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
    return model
    """
    Builds a transformer model for music generation.
    
    Args:
        vocab_size (int): Size of the vocabulary.
        num_raags (int): Number of raags in your dataset.
        sequence_length (int): Length of the input sequence.
        embedding_dim (int, optional): Dimension of the embedding layer. Defaults to 64.
        intermediate_dim (int, optional): Hidden dimension of the transformer layer. Defaults to 128.
        num_heads (int, optional): Number of attention heads. Defaults to 8.
        dropout (float, optional): Dropout rate. Defaults to 0.2.

    Returns:
        tf.keras.models.Model: A compiled Keras model.
    """