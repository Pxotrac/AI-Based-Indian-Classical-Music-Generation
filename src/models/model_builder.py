import tensorflow as tf
from tensorflow.keras import layers
import logging
from models.music_utils import get_token_frequencies, generate_random_seed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MusicTransformer(tf.keras.Model):
    """Music Transformer model."""

    def __init__(self, vocab_size, num_raags, sequence_length, embedding_dim=256, num_heads=8, ff_dim=512, rate=0.1):
        """
        Initializes the MusicTransformer model.

        Args:
            vocab_size (int): The size of the vocabulary.
            num_raags (int): The number of raags.
            sequence_length (int): The sequence length.
            embedding_dim (int, optional): The embedding dimension. Defaults to 256.
            num_heads (int, optional): The number of attention heads. Defaults to 8.
            ff_dim (int, optional): The dimension of the feedforward network. Defaults to 512.
            rate (float, optional): The dropout rate. Defaults to 0.1.
        """
        super(MusicTransformer, self).__init__()
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.num_raags = num_raags

        self.embedding = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)
        self.pos_encoding = PositionalEncoding(sequence_length, embedding_dim)
        self.transformer_layer = TransformerBlock(embedding_dim, num_heads, ff_dim, rate)
        self.dropout_1 = layers.Dropout(rate)
        self.dense = layers.Dense(vocab_size)
        
        # Raag Embedding
        self.raag_embedding = layers.Embedding(input_dim=num_raags, output_dim=embedding_dim)

    def call(self, inputs, training):
        """
        Forward pass of the MusicTransformer model.

        Args:
            inputs (tuple): A tuple containing the notes input and the raag labels.
            training (bool): Whether the model is in training mode.

        Returns:
            tf.Tensor: The output tensor.
        """
        notes_input, raag_labels = inputs
        logging.debug(f"Notes_input shape: {notes_input.shape}")
        logging.debug(f"Raag_labels shape: {raag_labels.shape}")
        
        # Notes embedding
        x = self.embedding(notes_input)
        x *= tf.math.sqrt(tf.cast(self.embedding_dim, tf.float32))
        x = self.pos_encoding(x)
        x = self.dropout_1(x, training=training)

        x = self.transformer_layer(x, training=training)
        
        # Raag Conditioning
        raag_embeddings = self.raag_embedding(raag_labels) # [batch_size, embedding_dim] 
        raag_embeddings = tf.expand_dims(raag_embeddings, axis=1) # [batch_size, 1, embedding_dim]
        raag_embeddings = tf.tile(raag_embeddings, [1, self.sequence_length, 1]) # [batch_size, sequence_length, embedding_dim]

        # concatenate
        x = tf.concat([x, raag_embeddings], axis=-1) # [batch_size, sequence_length, embedding_dim*2]

        # Output Layer
        x = self.dense(x) # [batch_size, sequence_length, vocab_size]
        x = x[:, -1, :] # we only take the last prediction

        return x  # [batch_size, vocab_size]

class PositionalEncoding(tf.keras.layers.Layer):
    """Positional encoding layer."""

    def __init__(self, position, d_model):
        """
        Initializes the PositionalEncoding layer.

        Args:
            position (int): The maximum position.
            d_model (int): The model dimension.
        """
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def call(self, x):
        """
        Forward pass of the PositionalEncoding layer.

        Args:
            x (tf.Tensor): The input tensor.

        Returns:
            tf.Tensor: The output tensor with positional encoding.
        """
        length = tf.shape(x)[1]
        x += self.pos_encoding[:length, :]
        return x

    def get_angles(self, pos, i, d_model):
        """
        Calculates the angle for positional encoding.

        Args:
            pos (tf.Tensor): The position tensor.
            i (tf.Tensor): The index tensor.
            d_model (int): The model dimension.

        Returns:
            tf.Tensor: The angle tensor.
        """
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return pos * angles

    def positional_encoding(self, position, d_model):
        """
        Creates the positional encoding matrix.

        Args:
            position (int): The maximum position.
            d_model (int): The model dimension.

        Returns:
            tf.Tensor: The positional encoding matrix.
        """
        angle_rads = self.get_angles(
            pos=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model
        )
        # Apply sin to even indices in the array
        sines = tf.math.sin(angle_rads[:, 0::2])
        # Apply cos to odd indices in the array
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

class TransformerBlock(tf.keras.layers.Layer):
    """Transformer block layer."""

    def __init__(self, d_model, num_heads, ff_dim, rate=0.1):
        """
        Initializes the TransformerBlock layer.

        Args:
            d_model (int): The model dimension.
            num_heads (int): The number of attention heads.
            ff_dim (int): The dimension of the feedforward network.
            rate (float, optional): The dropout rate. Defaults to 0.1.
        """
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(d_model),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        """
        Forward pass of the TransformerBlock layer.

        Args:
            inputs (tf.Tensor): The input tensor.
            training (bool): Whether the model is in training mode.

        Returns:
            tf.Tensor: The output tensor.
        """
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def create_model(vocab_size, num_raags, sequence_length, strategy):
    """
    Creates the MusicTransformer model.

    Args:
        vocab_size (int): The vocabulary size.
        num_raags (int): The number of raags.
        sequence_length (int): The sequence length.
        strategy (tf.distribute.Strategy): The distribution strategy.

    Returns:
        tf.keras.Model: The created model.
    """
    with strategy.scope():
        logging.info("Creating model...")
        notes_input = tf.keras.Input(shape=(sequence_length,), name="notes_input")
        raag_label = tf.keras.Input(shape=(), name="raag_label", dtype=tf.int32) #modified
        
        transformer_output = MusicTransformer(vocab_size, num_raags, sequence_length)([notes_input, raag_label])
        # Add a Dense layer with softmax activation for the output
        output = transformer_output #removed dense, because it is already in MusicTransformer.

        model = tf.keras.Model(inputs=[notes_input, raag_label], outputs=output, name='music_transformer')
        return model
