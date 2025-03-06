import pickle
import os
import logging
from models.data_utils import create_tokenizer, load_and_preprocess_data, extract_all_notes

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Tests the tokenizer creation process.
    """
    tokenizer_name = "transformer_tokenizer.pickle"
    # Check if running on Colab and set repo_dir accordingly
    if os.environ.get("COLAB_GPU", "FALSE") == "TRUE":
        repo_dir = "/content/drive/MyDrive/music_generation_repo"
        data_path = "/content/drive/MyDrive/"
        print(f"Running on Colab. repo_dir: {repo_dir}")
        print(f"Running on Colab. data_path: {data_path}")
    else:
        repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.dirname(repo_dir)
        print(f"Running locally. repo_dir: {repo_dir}")
        print(f"Running locally. data_path: {data_path}")

    try:
        all_output = load_and_preprocess_data(repo_dir, data_path)
        if all_output is None or len(all_output) == 0:
          logging.error("No data was found. Aborting")
          return
        all_notes = extract_all_notes(all_output)
        all_notes_flatten = [item for sublist in all_notes for item in sublist]
        tokenizer = create_tokenizer(all_notes_flatten)
        if tokenizer is None:
          logging.error("Tokenizer was not created. Check preprocessing. Aborting")
          return
        if len(tokenizer.word_index) < 2:
            logging.error("Tokenizer created a vocabulary with less than 2 words. Check your notes data. Aborting.")
            return
        vocab_size = len(tokenizer.word_index) + 1
        logging.info(f"Tokenizer created successfully!")
        logging.info(f"Vocabulary size: {vocab_size}")
        logging.info(f"First 5 words in the vocabulary: {list(tokenizer.word_index.items())[:5]}")
        logging.info(f"Loaded {len(all_output)} data samples.")
        for i, item in enumerate(all_output):
          logging.info(f"Data sample {i}: {item}")
          if i >= 5:
            break

    except Exception as e:
        logging.error(f"Error creating the tokenizer: {e}")

if __name__ == "__main__":
    main()
