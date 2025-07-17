import numpy as np
import pickle
import os # For checking file existence in load_index

# --- Helper Function: Cosine Similarity ---
def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculates the cosine similarity between two 1D NumPy arrays (vectors).

    Args:
        vec1 (np.ndarray): The first vector.
        vec2 (np.ndarray): The second vector.

    Returns:
        float: The cosine similarity score, ranging from -1.0 to 1.0.
    """
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0

    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    # Add a small epsilon to avoid division by zero if norm is extremely close to zero
    # This is rare for actual embeddings but good for robustness
    denominator = norm_vec1 * norm_vec2
    if denominator < 1e-9: # A very small number close to zero
        return 0.0

    similarity = dot_product / denominator
    return float(similarity)


class VectorDatabase:
    """
    A simple in-memory vector database with Brute-Force and LSH search capabilities.
    """
    def __init__(self, num_hyperplanes_per_table: int = 32, num_hash_tables: int = 16):
        """
        Initializes an empty vector database and LSH index parameters.

        Args:
            num_hyperplanes_per_table (int): K - Number of random hyperplanes per hash table.
            num_hash_tables (int): L - Number of independent hash tables.
        """
        self.texts = []
        self.embeddings = None
        self.embedding_dim = None

        # LSH parameters
        self.num_hyperplanes_per_table = num_hyperplanes_per_table
        self.num_hash_tables = num_hash_tables
        self.hyperplanes = [] # List of (num_hyperplanes_per_table, embedding_dim) arrays for each table
        self.lsh_indexes = [] # List of dictionaries, each representing a hash table

        print("Vector database initialized with LSH parameters.")

    def _generate_hyperplanes(self, embedding_dim: int):
        """
        Generates random hyperplanes for LSH. Called once when the first embedding is added.
        Each hyperplane is a random vector with components from a standard normal distribution.
        """
        if self.hyperplanes:
            return

        print(f"Generating {self.num_hash_tables} sets of {self.num_hyperplanes_per_table} hyperplanes (dim {embedding_dim})...")
        for _ in range(self.num_hash_tables):
            hyperplane_set = np.random.randn(self.num_hyperplanes_per_table, embedding_dim)
            self.hyperplanes.append(hyperplane_set)
            self.lsh_indexes.append({})
        print("Hyperplanes generated.")

    def _get_hash_code(self, embedding: np.ndarray, hyperplane_set: np.ndarray) -> str:
        """
        Generates a binary hash code for a given embedding using a set of hyperplanes.
        The hash bit is 1 if the dot product is positive, 0 otherwise.

        Args:
            embedding (np.ndarray): The 1D embedding vector.
            hyperplane_set (np.ndarray): A 2D array of hyperplane vectors for one hash table.

        Returns:
            str: The binary hash code (e.g., '01101...').
        """
        dot_products = np.dot(hyperplane_set, embedding)
        hash_bits = (dot_products >= 0).astype(int)
        return "".join(map(str, hash_bits))


    def add_vectors(self, texts: list[str], embeddings: np.ndarray):
        """
        Adds new texts and their corresponding embeddings to the database and updates LSH index.

        Args:
            texts (list[str]): A list of original text strings.
            embeddings (np.ndarray): A NumPy array of embeddings,
                                     where embeddings[i] corresponds to texts[i].
                                     Shape should be (num_new_texts, embedding_dimension).
        """
        if not texts or embeddings.size == 0:
            print("No data to add.")
            return

        if len(texts) != embeddings.shape[0]:
            raise ValueError("Number of texts must match the number of embeddings.")

        new_vector_start_idx = len(self.texts)

        self.texts.extend(texts)

        if self.embeddings is None:
            self.embeddings = embeddings
            self.embedding_dim = embeddings.shape[1]
            self._generate_hyperplanes(self.embedding_dim)
        else:
            if self.embedding_dim != embeddings.shape[1]:
                raise ValueError("New embeddings must have the same dimension as existing ones.")
            self.embeddings = np.vstack((self.embeddings, embeddings))

        for i, new_embedding in enumerate(embeddings):
            global_index = new_vector_start_idx + i
            for table_idx, hyperplane_set in enumerate(self.hyperplanes):
                hash_code = self._get_hash_code(new_embedding, hyperplane_set)
                if hash_code not in self.lsh_indexes[table_idx]:
                    self.lsh_indexes[table_idx][hash_code] = []
                self.lsh_indexes[table_idx][hash_code].append(global_index)

        print(f"Added {len(texts)} new vectors to the database and updated LSH index.")
        print(f"Total vectors in database: {len(self.texts)}")

    def brute_force_search(self, query_text: str, k: int, embedding_generator) -> list[tuple[str, float]]:
        """
        Performs a brute-force (linear) search to find the top-k most similar items
        to a given query text.

        Args:
            query_text (str): The text to query for similarity.
            k (int): The number of top similar items to retrieve.
            embedding_generator (EmbeddingGenerator): An instance of the EmbeddingGenerator
                                                     to convert query_text to an embedding.

        Returns:
            list[tuple[str, float]]: A list of (text, similarity_score) tuples,
                                     sorted by similarity score in descending order.
        """
        if self.embeddings is None or self.embeddings.size == 0:
            print("Database is empty. No search can be performed.")
            return []

        query_embedding = embedding_generator.generate_embeddings([query_text])
        if query_embedding.size == 0:
            print("Could not generate embedding for query text.")
            return []
        query_embedding = query_embedding[0]

        similarities = []
        print(f"Performing brute-force search for '{query_text}' among {len(self.texts)} vectors...")
        for i, stored_embedding in enumerate(self.embeddings):
            similarity = calculate_cosine_similarity(query_embedding, stored_embedding)
            similarities.append((self.texts[i], similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        print("Brute-force search completed.")
        return similarities[:k]

    def lsh_search(self, query_text: str, k: int, embedding_generator) -> list[tuple[str, float]]:
        """
        Performs an Approximate Nearest Neighbor (ANN) search using LSH to find
        the top-k most similar items to a given query text.

        Args:
            query_text (str): The text to query for similarity.
            k (int): The number of top similar items to retrieve.
            embedding_generator (EmbeddingGenerator): An instance of the EmbeddingGenerator
                                                     to convert query_text to an embedding.

        Returns:
            list[tuple[str, float]]: A list of (text, similarity_score) tuples,
                                     sorted by similarity score in descending order.
        """
        if self.embeddings is None or self.embeddings.size == 0:
            print("Database is empty. No search can be performed.")
            return []
        if not self.lsh_indexes:
            print("LSH index not built. Please add vectors first.")
            return []

        query_embedding = embedding_generator.generate_embeddings([query_text])
        if query_embedding.size == 0:
            print("Could not generate embedding for query text.")
            return []
        query_embedding = query_embedding[0]

        candidate_indices = set()
        print(f"Collecting candidates for LSH search for '{query_text}'...")
        for table_idx, hyperplane_set in enumerate(self.hyperplanes):
            hash_code = self._get_hash_code(query_embedding, hyperplane_set)
            if hash_code in self.lsh_indexes[table_idx]:
                candidate_indices.update(self.lsh_indexes[table_idx][hash_code])
        print(f"Collected {len(candidate_indices)} unique candidates from LSH index.")

        if not candidate_indices:
            print("No candidates found in LSH buckets for the query.")
            return []

        similarities = []
        for idx in candidate_indices:
            stored_embedding = self.embeddings[idx]
            similarity = calculate_cosine_similarity(query_embedding, stored_embedding)
            similarities.append((self.texts[idx], similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        print("LSH search completed.")
        return similarities[:k]

    def save_index(self, filepath: str):
        """
        Saves the entire VectorDatabase instance (including its LSH index,
        embeddings, and texts) to a file using pickle.

        Args:
            filepath (str): The path to the file where the index will be saved.
        """
        print(f"Saving index to {filepath}...")
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            print("Index saved successfully.")
        except Exception as e:
            print(f"Error saving index: {e}")

    @staticmethod
    def load_index(filepath: str) -> 'VectorDatabase':
        """
        Loads a VectorDatabase instance from a file using pickle.

        Args:
            filepath (str): The path to the file from which the index will be loaded.

        Returns:
            VectorDatabase: The loaded VectorDatabase instance.
        Raises:
            FileNotFoundError: If the specified file does not exist.
            Exception: For other errors during loading.
        """
        if not os.path.exists(filepath):
            print(f"Error: Index file not found at {filepath}. Cannot load.")
            raise FileNotFoundError(f"Index file not found: {filepath}")

        print(f"Loading index from {filepath}...")
        try:
            with open(filepath, 'rb') as f:
                loaded_db = pickle.load(f)
            print("Index loaded successfully.")
            return loaded_db
        except Exception as e:
            print(f"Error loading index: {e}")
            raise