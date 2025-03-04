import pickle
import os
import logging
from models.data_utils import create_tokenizer, load_and_preprocess_data, extract_all_notes

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Tests the tokenizer creation process.

    This function attempts to create a tokenizer using the `create_tokenizer`
    function from `data_utils.py`. It then checks if the tokenizer was
    created successfully and prints the vocabulary size and the first
    five words of the vocabulary.
    """
    tokenizer_name = "transformer_tokenizer.pickle"  # Specify the file name
    # Check if running on Colab and set repo_dir accordingly
    if os.environ.get("COLAB_GPU", "FALSE") == "TRUE":
        repo_dir = "/content/drive/MyDrive/music_generation_repo"
        data_path = "/content/drive/MyDrive/"  #correct path
        print(f"Running on Colab. repo_dir: {repo_dir}")
        print(f"Running on Colab. data_path: {data_path}")
    else:
        repo_dir = os.path.dirname(os.path.abspath(__file__))
        repo_dir = os.path.dirname(repo_dir)  # Go up one more level
        data_path = os.path.dirname(repo_dir)
        print(f"Running locally. repo_dir: {repo_dir}")
        print(f"Running locally. data_path: {data_path}")

    try:
        all_output = load_and_preprocess_data(repo_dir, data_path) #change
        if all_output is None or len(all_output) == 0:
          logging.error("No data was found. Aborting")
          return
        all_notes = extract_all_notes(all_output)
        tokenizer = create_tokenizer(all_notes)
        if tokenizer is None:
          logging.error("Tokenizer was not created. Check preprocessing. Aborting")
          return
        vocab_size = len(tokenizer.word_index) + 1
        logging.info(f"Tokenizer created successfully!")
        logging.info(f"Vocabulary size: {vocab_size}")  # Check the number of tokens
        logging.info(f"First 5 words in the vocabulary {list(tokenizer.word_index.items())[:5]}")

    except Exception as e:
        logging.error(f"Error creating the tokenizer: {e}")

if __name__ == "__main__":
    main()
