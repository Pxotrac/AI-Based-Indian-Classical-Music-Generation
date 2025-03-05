import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, MultiHeadAttention, LayerNormalization, Dense, Dropout, Layer
from tensorflow.keras.models import Model
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RaagConditioning(Layer):
    def __init__(self, num_raags, embedding_dim, sequence_length):
        super(RaagConditioning, self).__init__()
        self.raag_embedding = Embedding(input_dim=num_raags, output_dim=embedding_dim)
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim  # Store embedding_dim

    def call(self, raag_embeddings):
        # Look up the embedding for the raag ID
        raag_embed = self.raag_embedding(raag_embeddings)
        # Tile the raag embedding to match the sequence length
        return tf.tile(raag_embed, [1, self.sequence_length, 1])

class TransformerEncoderLayer(Layer):
    def __init__(self, embedding_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embedding_dim)  # Keep the output dimension consistent with the input
        ])
        
        self.layernorm1 = LayerNormalization(epsilon=1e-6) 
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.mha(inputs, inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output) # Modified
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class MusicTransformer(Model):
    def __init__(self, vocab_size, num_raags, sequence_length, embedding_dim=256, num_heads=4, ff_dim=512, rate=0.1, strategy=None):
        super(MusicTransformer, self).__init__()
        self.embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)
        self.raag_conditioning = RaagConditioning(num_raags, embedding_dim, sequence_length)
        self.encoder_layer = TransformerEncoderLayer(embedding_dim, num_heads, ff_dim, rate)
        self.dropout = Dropout(rate)
        self.dense = Dense(vocab_size, activation='softmax')
        self.sequence_length = sequence_length
    
    def call(self, inputs, training):
        """
       Defines the forward pass of the MusicTransformer model.
   
       Args:
           inputs: A tuple containing the sequence input and the raag input.
           training: A boolean indicating whether the model is in training mode.
       Returns:
           The output of the model, which is a probability distribution over the vocabulary.
       """
        sequence_input, raag_input = inputs
        #embedding
        x = self.embedding(sequence_input)
        x = self.dropout(x, training=training)
        #Raag conditioning.
        raag_embeddings = self.raag_conditioning(raag_input)  # Call RaagConditioning to get raag embeddings
        # Concatenate sequence embeddings and raag embeddings
        x = tf.concat([x, raag_embeddings], axis=-1)
        #encoder
        x = self.encoder_layer(x, training)
        x = self.dropout(x, training=training)
        #dense
        return self.dense(x)

def create_model(vocab_size, num_raags, sequence_length, strategy):
    """Creates the MusicTransformer model."""
    logging.info("Creating model...")
    with strategy.scope():
        input_sequence = Input(shape=(sequence_length,))
        input_raag = Input(shape=(1,)) #added input raag
        music_transformer = MusicTransformer(vocab_size, num_raags, sequence_length)
        output = music_transformer((input_sequence, input_raag)) #modified
        model = Model(inputs=[input_sequence, input_raag], outputs=output) #modified
        return model
