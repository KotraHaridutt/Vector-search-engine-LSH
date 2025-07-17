import time
import os # For os.path.exists if you use it in the main logic before loading
import numpy as np # Still needed for general numpy operations in main if any

from embedding_generator import EmbeddingGenerator
from vector_database import VectorDatabase

if __name__ == "__main__":
    embedding_generator = EmbeddingGenerator()

    # --- Configuration ---
    INDEX_FILE = "vector_db_index.pkl"
    # Increased num_replications for more significant speedup demonstration
    # For a practical demo, 10,000-50,000 vectors often show clear benefits.
    num_replications = 5000 # This will create 12 * 5000 = 60,000 vectors
    k_results = 5
    query_text_search = "Tell me about animals that are quick and lazy."

    # --- Scenario 1: Build, Save, and then Load the Index ---
    print("\n--- Scenario 1: Building and Saving Index ---")
    base_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast, ginger canine leaps over a lethargic hound.",
        "Artificial intelligence is transforming the world.",
        "The cat sat on the mat.",
        "Generative AI models are incredibly powerful.",
        "Machine learning is a subset of AI.",
        "Natural language processing deals with text data.",
        "Deep learning uses neural networks.",
        "Cloud computing provides scalable resources.",
        "Data science involves statistical analysis.",
        "The universe is vast and full of stars.",
        "Planets orbit around stars in galaxies."
    ]
    large_sample_texts = []
    for i in range(num_replications):
        for text in base_texts:
            large_sample_texts.append(f"{text} (rep {i})")

    # Check if index file exists, if so, load it directly
    if os.path.exists(INDEX_FILE):
        print(f"Index file '{INDEX_FILE}' found. Attempting to load existing index...")
        try:
            my_vector_db = VectorDatabase.load_index(INDEX_FILE)
            print(f"Loaded database contains {len(my_vector_db.texts)} vectors.")
        except Exception as e:
            print(f"Failed to load existing index: {e}. Rebuilding index.")
            # If loading fails, proceed to build a new one
            my_vector_db = VectorDatabase(num_hyperplanes_per_table=8, num_hash_tables=64)
            my_vector_db.add_vectors(large_sample_texts, large_embeddings)
            my_vector_db.save_index(INDEX_FILE)
    else:
        print(f"Index file '{INDEX_FILE}' not found. Building new index...")
        large_embeddings = embedding_generator.generate_embeddings(large_sample_texts)
        my_vector_db = VectorDatabase(num_hyperplanes_per_table=8, num_hash_tables=64)
        my_vector_db.add_vectors(large_sample_texts, large_embeddings)
        my_vector_db.save_index(INDEX_FILE)


    # --- Perform searches on the (either built or loaded) database ---
    print(f"\n--- Performing Brute-Force Search for '{query_text_search}' (k={k_results}) ---")
    start_time_brute_force = time.perf_counter()
    top_k_brute_force = my_vector_db.brute_force_search(query_text_search, k_results, embedding_generator)
    end_time_brute_force = time.perf_counter()
    brute_force_time = end_time_brute_force - start_time_brute_force

    print("\nBrute-Force Search Results:")
    if top_k_brute_force:
        for i, (text, score) in enumerate(top_k_brute_force):
            print(f"  {i+1}. Score: {score:.4f} - Text: '{text}'")
    else:
        print("  No results found.")
    print(f"\nBrute-Force Search Time: {brute_force_time:.6f} seconds")


    print(f"\n--- Performing LSH Search for '{query_text_search}' (k={k_results}) ---")
    start_time_lsh = time.perf_counter()
    top_k_lsh = my_vector_db.lsh_search(query_text_search, k_results, embedding_generator)
    end_time_lsh = time.perf_counter()
    lsh_time = end_time_lsh - start_time_lsh

    print("\nLSH Search Results:")
    if top_k_lsh:
        for i, (text, score) in enumerate(top_k_lsh):
            print(f"  {i+1}. Score: {score:.4f} - Text: '{text}'")
    else:
        print("  No results found.")
    print(f"\nLSH Search Time: {lsh_time:.6f} seconds")

    # --- Performance Comparison ---
    if brute_force_time > 0 and lsh_time > 0:
        speedup = brute_force_time / lsh_time
        print(f"\n--- Performance Comparison ---")
        print(f"Brute-Force Time: {brute_force_time:.6f}s")
        print(f"LSH Time:         {lsh_time:.6f}s")
        print(f"LSH is {speedup:.2f}x faster than Brute-Force (for this query and dataset size).")
    else:
        print("\nCannot compare performance (one or both times were zero).")

    # --- Accuracy Check (Optional but Recommended for Demo) ---
    print("\n--- Accuracy Check (LSH vs Brute-Force) ---")
    # Compare top-k results
    brute_force_texts = [text for text, _ in top_k_brute_force]
    lsh_texts = [text for text, _ in top_k_lsh]

    common_results = set(brute_force_texts).intersection(set(lsh_texts))
    recall_lsh = len(common_results) / len(brute_force_texts) if brute_force_texts else 0

    print(f"Top {k_results} results from Brute-Force: {brute_force_texts}")
    print(f"Top {k_results} results from LSH: {lsh_texts}")
    print(f"Number of common results: {len(common_results)}")
    print(f"Recall of LSH (overlap with Brute-Force): {recall_lsh:.2%}")
    if recall_lsh < 1.0:
        print("Note: LSH is an approximate algorithm and may not return identical results to brute-force.")