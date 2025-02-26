import pickle
from tensorflow.keras.preprocessing.text import Tokenizer

try:
    # Load the tokenizer
    with open("ragatokenzier.pkl", "rb") as f:
        tokenizer = pickle.load(f)
        print("Tokenizer loaded successfully!")
        print("Vocabulary size:", len(tokenizer.word_index))  # Check the number of tokens
        print("First 5 words in the vocabulary", list(tokenizer.word_index.items())[:5])
except Exception as e:
    print(f"Error loading tokenizer: {e}")