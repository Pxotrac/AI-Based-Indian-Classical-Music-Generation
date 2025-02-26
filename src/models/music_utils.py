import numpy as np
import pretty_midi
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_music(model, seed_sequence, raag_id, max_length, temperature=1.0, top_k=40, token_frequencies=None):
    """Generates music using top-k sampling and token frequency balancing."""
    generated_sequence = seed_sequence.copy()
    for _ in range(max_length - len(seed_sequence)):
        input_sequence = np.array([generated_sequence[-sequence_length:]])
        raag_input = np.array([[raag_id]])

        # Predict the next note
        prediction = model.predict([input_sequence, raag_input], verbose=0)

        # Apply temperature scaling
        prediction = prediction / temperature

        # Top-k Sampling
        top_k = min(top_k, vocab_size)
        indices = np.argpartition(prediction[0], -top_k)[-top_k:]
        probabilities = prediction[0][indices]
        probabilities = probabilities / np.sum(probabilities)

        # Token Frequency Balancing (if token_frequencies is provided)
        if token_frequencies:
            for i, index in enumerate(indices):
                probabilities[i] /= token_frequencies.get(index, 1)
            probabilities = probabilities / np.sum(probabilities)

        # Sample from top-k tokens
        next_token = np.random.choice(indices, p=probabilities)

        generated_sequence.append(next_token)

    return generated_sequence

def tokens_to_midi(generated_tokens, tokenizer, tonic_hz=440):
    """Converts generated tokens to a MIDI file."""
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    instrument = pretty_midi.Instrument(program=0)

    current_time = 0
    for token in generated_tokens:
        note_name = tokenizer.index_word.get(token)

        # Convert note name to MIDI note number
        midi_note = None
        if note_name == "Sa":
            midi_note = 60
        elif note_name == "Re":
            midi_note = 62
        elif note_name == "Ga":
            midi_note = 64
        elif note_name == "Ma":
            midi_note = 65
        elif note_name == "Pa":
            midi_note = 67
        elif note_name == "Dha":
            midi_note = 69
        elif note_name == "Ni":
            midi_note = 71

        if midi_note is not None:
            note = pretty_midi.Note(velocity=100, pitch=midi_note, start=current_time, end=current_time + 0.5)
            instrument.notes.append(note)
            current_time += 0.5

    midi.instruments.append(instrument)
    return midi

def generate_raag_music(model, raag_id, seed_sequence, max_length=100, temperature=1.2, top_k=30):
    """Generates music for a specific raag, potentially adjusting tonic frequency."""

    tonic_frequencies = {
        # Add the tonic frequencies for each raag ID in Hz
        0: 440,  # Example: raag_id 0 has a tonic frequency of 440 Hz
        1: 432,  # Example: raag_id 1 has a tonic frequency of 432 Hz
        # ... (add mappings for other raag IDs)
    }
    tonic_hz = tonic_frequencies.get(raag_id, 440)  # Default to 440 if not found

    generated_tokens = generate_music(model, seed_sequence, raag_id, max_length, temperature, top_k)

    midi_data = tokens_to_midi(generated_tokens, tokenizer, tonic_hz=tonic_hz)
    midi_data.write(f"generated_music_raag_{raag_id}.mid")

    return generated_tokens

def generate_music_with_tonic(model, seed_sequence, raag_id, max_length, temperature=1.0, top_k=40, tonic_hz=440):
    """Generates music with raag-specific IDs and dynamic parameters."""
    logging.info("Generating music with raag-specific tonic...")
    generated_sequence = seed_sequence.copy()

    # Apply raag-specific tonic frequency shift if needed
    midi_mapping = {
        "Sa": 60, "Re": 62, "Ga": 64, "Ma": 65,
        "Pa": 67, "Dha": 69, "Ni": 71
    }

    # Adjust midi_mapping based on tonic_hz
    for note, midi_num in midi_mapping.items():
        midi_mapping[note] = midi_num + 12 * int(round(np.log2(tonic_hz / 440)))  # Shift by octaves based on tonic_hz

    for _ in range(max_length - len(seed_sequence)):
        input_sequence = np.array([generated_sequence[-sequence_length:]])
        raag_input = np.array([[raag_id]])

        # Predict the next note
        prediction = model.predict([input_sequence, raag_input], verbose=0)

        # Apply temperature scaling
        prediction = prediction / temperature

        # Top-k Sampling
        top_k = min(top_k, vocab_size)
        indices = np.argpartition(prediction[0], -top_k)[-top_k:]
        probabilities = prediction[0][indices]
        probabilities = probabilities / np.sum(probabilities)

        # Sample from top-k tokens
        next_token = np.random.choice(indices, p=probabilities)
        generated_sequence.append(next_token)

    # Convert tokens to MIDI
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    instrument = pretty_midi.Instrument(program=0)
    current_time = 0
    for token in generated_sequence:
        note_name = tokenizer.index_word.get(token)
        midi_note = midi_mapping.get(note_name)
        if midi_note is not None:
            note = pretty_midi.Note(velocity=100, pitch=midi_note, start=current_time, end=current_time + 0.5)
            instrument.notes.append(note)
            current_time += 0.5

    midi.instruments.append(instrument)
    midi.write(f"generated_music_raag_{raag_id}.mid")
    logging.info("Generated music and saved to MIDI")
    return generated_sequence

def generate_random_seed(tokenizer, sequence_length):
    """Generates a random seed sequence of specified length from the vocabulary of the tokenizer."""
    logging.info("Generating a random seed sequence")
    vocab = list(tokenizer.word_index.keys())  # Get all tokens from the vocabulary
    seed = np.random.choice(vocab, sequence_length).tolist()  # Randomly select tokens
    seed_tokens = [tokenizer.word_index[token] for token in seed]  # Convert to numerical tokens

    return seed_tokens

def get_token_frequencies(all_notes):
    """Calculates token frequencies from all_notes and normalizes them."""
    logging.info("Calculating token frequencies from all notes...")
    token_counts = Counter(all_notes)
    total_count = sum(token_counts.values())
    token_frequencies = {k: v / total_count for k, v in token_counts.items()}
    logging.info("Token frequencies calculated successfully")
    return token_frequencies