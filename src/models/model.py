import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Custom Layers ----------------------------------------------------------------
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention = tf.nn.softmax(
            (tf.matmul(q, k, transpose_b=True)) / tf.math.sqrt(tf.cast(self.depth, tf.float32))
        )
        output = tf.matmul(scaled_attention, v)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))
        return self.dense(output)

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):  # Add **kwargs
        super().__init__(**kwargs)  # Pass **kwargs to super().__init__
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(d_model, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        attn_output = self.attn(inputs, inputs, inputs)  # Self-attention
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + ffn_output)

class RaagConditioning(tf.keras.layers.Layer):
    def __init__(self, sequence_length, **kwargs):  # Add **kwargs
        super().__init__(**kwargs)  # Pass **kwargs to super().__init__
        self.sequence_length = sequence_length

    def call(self, raag_embed):
        return tf.tile(raag_embed, [1, self.sequence_length, 1])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.sequence_length, input_shape[2])
