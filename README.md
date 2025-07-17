# Vector-search-engine-LSH
# Simple Vector Database with LSH

## Project Overview
This project implements a simplified in-memory vector database, demonstrating core concepts of vector embeddings, similarity search, and Approximate Nearest Neighbor (ANN) indexing using Locality-Sensitive Hashing (LSH). It provides a foundational understanding of how modern similarity search systems work.

## Features
-   **Text Embedding Generation:** Utilizes the `sentence-transformers` library to convert text into high-dimensional numerical vector embeddings.
-   **Vector Storage:** Stores text-embedding pairs in memory.
-   **Brute-Force Similarity Search:** Implements a linear scan approach to find the exact nearest neighbors based on cosine similarity. Serves as a baseline for performance comparison.
-   **Locality-Sensitive Hashing (LSH) Indexing:** Builds an LSH index using random hyperplanes for efficient approximate nearest neighbor search.
-   **LSH Search:** Performs rapid approximate similarity searches by querying the LSH index, drastically reducing the search space.
-   **Persistence:** Ability to save and load the entire vector database (including the LSH index) to/from disk using Python's `pickle` module.
-   **Performance Benchmarking:** Measures and compares the search speed of brute-force vs. LSH.
-   **Accuracy Evaluation:** Evaluates the recall of the LSH algorithm against the brute-force baseline.

## Technical Stack
-   Python 3.13
-   `numpy`: For numerical operations and efficient vector handling.
-   `sentence-transformers`: For generating text embeddings.
-   `pickle`: For object serialization (persistence).

## Setup and Installation

1.  **Clone the repository (if you're a new user):**
    ```bash
    git clone KotraHaridutt/Vector-search-engine-LSH
    cd simple_vector_db
    ```
    (If you're already in your project directory, skip cloning.)

2.  **Create a Python Virtual Environment (recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the Virtual Environment:**
    -   On Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    -   On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

4.  **Install Dependencies:**
    Create a `requirements.txt` file at the root of your project with the following content:
    ```
    numpy
    sentence-transformers
    ```
    Then, install:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

1.  Ensure your virtual environment is active.
2.  Run the main script from your project's root directory:
    ```bash
    python main.py
    ```

The script will:
-   Load the embedding model.
-   (If `vector_db_index.pkl` does not exist) Generate a large dataset of embeddings and build the LSH index.
-   (If `vector_db_index.pkl` exists) Load the existing index from disk.
-   Perform and benchmark both brute-force and LSH searches for a sample query.
-   Print results, search times, speedup, and LSH recall.

## Performance Analysis & LSH Tuning
(You will fill this section with your actual results. Below is an example.)

The `num_replications` variable in `main.py` controls the dataset size. As the dataset grows, the performance difference between brute-force and LSH becomes dramatic.

| Dataset Size (Vectors) | Brute-Force Time (s) | LSH Time (s) | Speedup (x) | LSH Recall (%) |
| :--------------------- | :------------------- | :----------- | :---------- | :------------- |
| 12,000                 | 0.045                | 0.002        | ~22.5       | 100            |
| 60,000                 | 0.220                | 0.005        | ~44.0       | 90             |
| ... (add more of your test results) | | | | |

LSH is an **approximate** algorithm. There's a trade-off between search speed and recall (how many of the true nearest neighbors are found).
-   `num_hyperplanes_per_table` (K): Controls the specificity of hash codes. Lower K generally means higher recall but more candidates to check.
-   `num_hash_tables` (L): Increases the number of independent "chances" to find a match, directly improving recall at the cost of more memory and slightly longer indexing/query time.
-   *Experimentation with K and L (e.g., K=8 to 32, L=16 to 64) is crucial to find the optimal balance for your specific dataset and requirements.*

## Future Enhancements
-   Implement other ANN algorithms (e.g., K-D Trees, Annoy, FAISS).
-   Support different similarity metrics (e.g., Euclidean distance).
-   Add support for concurrent queries.
-   Implement a more robust, production-ready persistence layer (e.g., using a dedicated vector database, or more efficient serialization for very large datasets).
-   Add deletion and update capabilities for vectors.
-   Build a simple API wrapper (e.g., with Flask or FastAPI).

## Author
KotraHaridutt
