import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingGenerator:
    """
    A class to handle the generation of vector embeddings from text.
    Uses a pre-trained sentence-transformer model.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initializes the EmbeddingGenerator by loading a pre-trained model.
        Args:
            model_name (str): The name of the sentence-transformer model to use.
                              'all-MiniLM-L6-v2' is a good balance of size and performance.
        """
        print(f"Loading embedding model: {model_name}...")
        try:
            self.model = SentenceTransformer(model_name)
            print("Embedding model loaded successfully.")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            print("Please ensure you have an internet connection or the model is cached.")
            raise

    def generate_embeddings(self, texts: list[str]) -> np.ndarray:
        """
        Generates embeddings for a list of input texts.

        Args:
            texts (list[str]): A list of strings (sentences or short documents).

        Returns:
            np.ndarray: A NumPy array where each row is the embedding for the corresponding text.
                        The shape will be (num_texts, embedding_dimension).
        """
        if not texts:
            return np.array([])
        # print(f"Generating embeddings for {len(texts)} texts...") # Optional: remove for cleaner output during large data processing
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        # print("Embeddings generated.") # Optional: remove for cleaner output
        return embeddings