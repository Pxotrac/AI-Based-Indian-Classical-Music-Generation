import numpy as np
import pretty_midi
import IPython.display as ipd

def generate_music(model, seed_sequence, raag_id, vocab_size, sequence_length, tokenizer, max_length=200, temperature=1.8, top_k=9):
    """Generates music using the model, incorporating temperature and top-k sampling."""
    generated = seed_sequence.copy()  # Initialize generated sequence
    for _ in range(max_length):
        # Trim sequence to the last `sequence_length` tokens
        input_seq = np.array([generated[-sequence_length:]])

        # Reshape input_seq to match the expected input shape of the model
        input_seq = input_seq.reshape(1, sequence_length)  # Reshape to (1, sequence_length)

        # Predict next token probabilities
        pred = model.predict([input_seq, np.array([[raag_id]])], verbose=0)[0]

        # Apply temperature scaling
        pred = pred / temperature
        pred = np.exp(pred) / np.sum(np.exp(pred))  # Softmax

        # Top-k sampling
        top_k_indices = np.argsort(pred)[-top_k:]
        top_k_probs = pred[top_k_indices]
        top_k_probs = top_k_probs / np.sum(top_k_probs)  # Renormalize

        # Sample from top-k
        next_token = np.random.choice(top_k_indices, p=top_k_probs)
        generated.append(next_token)

    return generated

def notes_to_midi(notes, output_path="generated_raga.mid", instrument_program=104):
    """Converts a sequence of musical notes to a MIDI file."""
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=instrument_program)
    midi.instruments.append(instrument)

    start_time = 0.0
    duration = 0.5  # Adjust based on desired tempo

    midi_mapping = {
        "Sa": 60, "Re": 62, "Ga": 64,
        "Ma": 65, "Pa": 67, "Dha": 69, "Ni": 71
    }

    for note_name in notes:
        midi_number = midi_mapping.get(note_name, 60)
        note = pretty_midi.Note(
            velocity=100,
            pitch=midi_number,
            start=start_time,
            end=start_time + duration
        )
        instrument.notes.append(note)
        start_time += duration  # Move to next beat

    midi.write(output_path)
    return output_path

def display_audio(midi_path):
    """Use ipd.Audio to display the audio (if available) or provide a link to download"""
    try:
        ipd.display(ipd.Audio(midi_path))
    except Exception as e:
        print(f"Could not play audio directly: {e}")
        print(f"Download the MIDI file here: {midi_path}")