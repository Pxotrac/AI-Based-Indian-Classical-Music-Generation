import numpy as np
import pretty_midi
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_music(model, seed_sequence, raag_id, max_length, temperature=1.0, top_k=40, token_frequencies=None, vocab_size=None, sequence_length=None):
    """Generates music using top-k sampling and token frequency balancing."""
    
    # Check if vocab_size and sequence_length are provided
    if vocab_size is None or sequence_length is None:
        logging.error("vocab_size and sequence_length must be provided to generate_music")
        return []

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
                probabilities[i] /= token_frequencies.get(index, 1)  # Get frequency or default to 1
            probabilities = probabilities / np.sum(probabilities)  # Renormalize

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

def generate_raag_music(model, raag_id, seed_sequence, tokenizer, max_length=100, temperature=1.2, top_k=30, tonic_hz=440):
    """Generates music for a specific raag, potentially adjusting tonic frequency."""

    tonic_frequencies = {
        0: 349.23,  # Raag Bahar
        1: 440.00,  # Raag Bhairavi Bhajan
        2: 293.66,  # Raag Bhairavi Dadra
        3: 392.00,  # Raag Bhairavi Thumri
        4: 440.00,  # Raag Bhimpalasi
        5: 293.66,  # Raag Bibhas
        6: 440.00,  # Raag Chandrakauns
        7: 293.66,  # Raag Deepki
        8: 329.63,  # Raag Dhani
        9: 440.00,  # Raag Gaud Malhar
        10: 293.66,  # Raag Gauri
        11: 349.23,  # Raag Gavti
        12: 440.00,  # Raag Hameer
        13: 392.00,  # Raag Khamaj
        14: 329.63,  # Raag Kirwani Bhajan
        15: 329.63,  # Raag Lalit Pancham
        16: 349.23,  # Raag Majh Khamaj Thumri
        17: 293.66,  # Raag Megh
        18: 392.00,  # Raag Multani
        19: 329.63,  # Raag Nirgun Bhajan
        20: 293.66,  # Raag Paraj
        21: 349.23,  # Raag Abhogi
        22: 392.00,  # Raag Ahir Bhairon
        23: 440.00,  # Raag Bageshree
        24: 261.63,  # Raag Bahar
        25: 493.88,  # Raag Bairagi
        26: 440.00,  # Raag Basanti Kedar
        27: 261.63,  # Raag Bhairav
        28: 392.00,  # Raag Bhatiyar
        29: 293.66,  # Raag Bhimpalasi
        30: 329.63,  # Raag Bhoopali
        31: 349.23,  # Raag Bibhas
        32: 392.00,  # Raag Bihag
        33: 261.63,  # Raag Bilaskhani Todi
        34: 349.23,  # Raag Chandrakauns
        35: 440.00,  # Raag Dagori
        36: 293.66,  # Raag Desh
        37: 261.63,  # Raag Dhani
        38: 349.23,  # Raag Gaud Malhar
        39: 440.00,  # Raag Gavti
        40: 329.63,  # Raag Hameer
        41: 493.88,  # Raag Hindol Pancham
        42: 261.63,  # Raag Jait Kalyan
        43: 349.23,  # Raag Jog
        44: 440.00,  # Raag Jogiya
        45: 392.00,  # Raag Kalyan
        46: 261.63,  # Raag Kedar
        47: 293.66,  # Raag Khamaj
        48: 349.23,  # Raag Khat Todi
        49: 493.88,  # Raag Khokar
        50: 392.00,  # Raag Komal Rishav Aasavari
        51: 440.00,  # Raag Lagan Gandhar
        52: 261.63,  # Raag Lalit
        53: 293.66,  # Raag Lalit Pancham
        54: 440.00,  # Raag Lalita Gauri
        55: 329.63,  # Raag Madhukauns
        56: 349.23,  # Raag Malkauns
        57: 392.00,  # Raag Marwa
        58: 440.00,  # Raag Megh
        59: 261.63,  # Raag Mian Malhar
        60: 329.63,  # Raag Miyan Malhar
        61: 440.00,  # Raag Multani
        62: 293.66,  # Raag Nat Kamod
        63: 349.23,  # Raag Paraj
        64: 493.88,  # Raag Poorva
        65: 440.00,  # Raag Puriya
        66: 392.00,  # Raag Puriya Dhanashree
        67: 261.63,  # Raag Rageshri
        68: 293.66,  # Raag Ramdasi Malhar
        69: 329.63,  # Raag Ramgauri
        70: 349.23,  # Raag Saraswati
        71: 440.00,  # Raag Sawani
        72: 493.88,  # Raag Shree
        73: 329.63,  # Raag Sohani
        74: 261.63,  # Raag Sooha Kanada
        75: 392.00,  # Raag Todi
        76: 349.23,  # Raag Triveni
        77: 261.63,  # Raag Yaman
        78: 293.66,  # Raageshree
        79: 349.23,  # Rageshree
        80: 392.00,  # Ramdasi Malhar
        81: 440.00,  # Shree
        82: 261.63,  # Sohani
        83: 293.66,  # Todi
        84: 329.63,  # Yaman
    }
    tonic_hz = tonic_frequencies.get(raag_id, 440)  # Default to 440 if not found

    generated_tokens = generate_music(model, seed_sequence, raag_id, max_length, temperature, top_k)
    
    midi_data = tokens_to_midi(generated_tokens, tokenizer, tonic_hz=tonic_hz)  # Pass tonic_hz to tokens_to_midi
    midi_data.write(f"generated_music_raag_{raag_id}.mid")

    return generated_tokens

def generate_music_with_tonic(model, seed_sequence, raag_id, tokenizer, max_length, temperature=1.0, top_k=40, tonic_hz=440, vocab_size=None, sequence_length=None):
    """Generates music with raag-specific IDs and dynamic parameters."""
    logging.info("Generating music with raag-specific tonic...")
    generated_sequence = seed_sequence.copy()

    # Check if vocab_size and sequence_length are provided
    if vocab_size is None or sequence_length is None:
        logging.error("vocab_size and sequence_length must be provided to generate_music_with_tonic")
        return []
    
    # Apply raag-specific tonic frequency shift if needed
    midi_mapping = {
        "Sa": 60, "Re": 62, "Ga": 64, "Ma": 65,
        "Pa": 67, "Dha": 69, "Ni": 71
    }

    # Adjust midi_mapping based on tonic_hz
    for note, midi_num in midi_mapping.items():
        midi_mapping[note] = midi_num + 12 * int(round(np.log2(tonic_hz / 440)))  # Shift by octaves based on tonic_hz

    for _ in range(max_length - len(seed_sequence)):
        input_sequence = np.array([generated_sequence[-sequence_length:]])  # Adjust if needed
        raag_input = np.array([[raag_id]])

        # Predict the next note
        prediction = model.predict([input_sequence, raag_input], verbose=0)

        # Apply temperature scaling
        prediction = prediction / temperature

        # Top-k Sampling
        top_k = min(top_k, vocab_size)
        indices = np.argpartition(prediction[0], -top_k)[-top_k:]
        probabilities = prediction[0][indices]
        probabilities = probabilities / np.sum(probabilities)  # Normalize probabilities

        # Sample from top-k tokens
        next_token = np.random.choice(indices, p=probabilities)  
        generated_sequence.append(next_token)


# Convert generated sequence to MIDI
midi = pretty_midi.PrettyMIDI(initial_tempo=120)
instrument = pretty_midi.Instrument(program=0)
current_time = 0
for token in generated_sequence:
    note_name = tokenizer.index_word.get(token)  # Get note name from token
    if note_name in midi_mapping:  # Check if it's a valid note
        midi_note = midi_mapping[note_name]
        note = pretty_midi.Note(velocity=100, pitch=midi_note, start=current_time, end=current_time + 0.5)
        instrument.notes.append(note)
        current_time += 0.5
midi.instruments.append(instrument)
midi.write(f"generated_music_raag_{raag_id}_with_tonic.mid")

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