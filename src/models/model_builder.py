import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Input, concatenate
from tensorflow.keras.models import Model

def create_model(vocab_size, num_raags, sequence_length, strategy):
    """Creates a music generation model with raag conditioning."""

    # Input layers
    notes_input = Input(shape=(sequence_length,), name='notes_input')
    raag_input = Input(shape=(1,), name='raag_input')  # Input for raag ID

    # Embedding layer for notes
    embedding_layer = Embedding(vocab_size, 256, input_length=sequence_length)(notes_input)

    # Embedding layer for raags
    raag_embedding_layer = Embedding(num_raags, 32)(raag_input)  # Embed raag ID
    raag_embedding_layer = tf.keras.layers.Flatten()(raag_embedding_layer)  # Flatten raag embedding

    # Concatenate note and raag embeddings
    merged_embeddings = concatenate([embedding_layer, tf.keras.layers.RepeatVector(sequence_length)(raag_embedding_layer)], axis=2)

    # LSTM layers
    lstm_layer1 = LSTM(512, return_sequences=True)(merged_embeddings)
    lstm_layer1 = Dropout(0.2)(lstm_layer1)
    lstm_layer2 = LSTM(512)(lstm_layer1)
    lstm_layer2 = Dropout(0.2)(lstm_layer2)

    # Dense output layer
    output_layer = Dense(vocab_size, activation='softmax')(lstm_layer2)

    # Create the model
    model = Model(inputs=[notes_input, raag_input], outputs=output_layer)

    # Compile the model
    optimizer = optimizers.legacy.Adam(learning_rate=0.001)  # Adjust learning rate if needed
    # optimizer = tf.tpu.experimental.CrossShardOptimizer(optimizer) if strategy.num_replicas_in_sync > 1 else optimizer # Wrap optimizer if using TPU strategy
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model