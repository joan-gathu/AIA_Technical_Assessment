"""
Document Processing Pipeline Module

This module provides functionality for preprocessing product review text, generating embeddings, and storing/retrieving them in ChromaDB for semantic search.

### Features:
- **Data Preprocessing** (`DataPreprocessor`): Cleans and normalizes text data by:
  - Lowercasing
  - Removing punctuation
  - Tokenization
  - Stopword removal
  - Lemmatization

- **Embedding Generation** (`EmbeddingGenerator`): Uses a pre-trained `SentenceTransformer` model to generate numerical embeddings for text.

- **ChromaDB Integration**:
  - `store_embeddings_in_chromadb()`: Stores review embeddings and metadata in ChromaDB.
  - `retrieve_similar_reviews()`: Retrieves the most relevant reviews based on semantic similarity.
"""

import os
import re
import string
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
import chromadb
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Ensure NLTK resources are downloaded only if missing
nltk_dependencies = ["stopwords", "punkt", "wordnet", "punkt_tab"]
for dep in nltk_dependencies:
    try:
        nltk.data.find(f"corpora/{dep}")
    except LookupError:
        nltk.download(dep)

class DataPreprocessor:
    """
    A class for preprocessing text data.

    Methods:
        clean_text(text): Cleans and normalizes text by removing stopwords, punctuation,
                          and applying lemmatization.
    """

    def __init__(self):
        """Initialize the DataPreprocessor with stopwords and a lemmatizer."""
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        """
        Preprocesses the input text by:
        - Converting to lowercase
        - Removing punctuation
        - Tokenizing
        - Removing stopwords
        - Lemmatizing
        """
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        words = word_tokenize(text)
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
        return " ".join(words)


class EmbeddingGenerator:
    """
    A class for generating embeddings from text using a pre-trained SentenceTransformer model.

    Uses all-MiniLM-L6-v2, a lightweight, high-performance embedding model.
    """

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize the embedding model."""
        self.model = SentenceTransformer(model_name)

    def generate_embedding(self, text):
        """Generate embeddings for the given text."""
        return self.model.encode(text, convert_to_tensor=True).tolist()


def store_embeddings_in_chromadb(df_reviews, db_path="./chroma_db", model_name="all-MiniLM-L6-v2"):
    """
    Stores review_text embeddings in ChromaDB for semantic search.

    Args:
    - df_reviews (pd.DataFrame): DataFrame containing review_text and metadata.
    - db_path (str): Path for persistent ChromaDB storage.
    - model_name (str): SentenceTransformer model to use for embeddings.

    Returns:
    - None (stores embeddings in ChromaDB)
    """

    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(path=db_path)
    collection = chroma_client.get_or_create_collection(name="product_reviews")

    # Load sentence transformer model
    model = SentenceTransformer(model_name)

    # Ensure unique review IDs
    if "review_id" not in df_reviews.columns:
        df_reviews = df_reviews.reset_index(drop=True)
        df_reviews["review_id"] = df_reviews.index.astype(str)

    # Encode reviews in batches for efficiency
    if "embedding" not in df_reviews.columns:
        logging.info("Generating embeddings for reviews...")
        df_reviews["embedding"] = model.encode(df_reviews["review_text"].tolist(), batch_size=32).tolist()

    # Store embeddings in ChromaDB
    for i, row in df_reviews.iterrows():
        try:
            collection.add(
                ids=[str(row["review_id"])],
                embeddings=[row["embedding"]],
                metadatas=[{
                    "review_text": row["review_text"],
                    "product": row["product"],
                    "category": row["category"],
                    "sentiment": row["sentiment"],
                    "rating": row["rating"]
                }]
            )
        except Exception as e:
            logging.error(f"Error storing review {row['review_id']}: {e}")

    logging.info("Successfully stored review embeddings in ChromaDB!")


def retrieve_similar_reviews(query, top_n=3, db_path="./chroma_db", model_name="all-MiniLM-L6-v2"):
    """
    Retrieves similar product reviews from ChromaDB based on a user query.

    Args:
    - query (str): User query to find relevant reviews.
    - top_n (int): Number of similar reviews to retrieve.
    - db_path (str): Path for persistent ChromaDB storage.
    - model_name (str): SentenceTransformer model for query embedding.

    Returns:
    - List of dictionaries containing the top matching reviews and metadata.
    """

    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(path=db_path)
    collection = chroma_client.get_or_create_collection(name="product_reviews")

    # Load embedding model
    model = SentenceTransformer(model_name)

    # Generate embedding for the query
    query_embedding = model.encode(query).tolist()

    logging.info(f"Searching for similar reviews to: {query}")

    # Search for the most relevant reviews
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_n
    )

    # Extract and return metadata of retrieved reviews
    if results["metadatas"]:
        logging.info(f"Found {len(results['metadatas'][0])} similar reviews.")
        return results["metadatas"][0]
    else:
        logging.warning("No similar reviews found.")
        return []



if __name__ == "__main__":
    # Use absolute path for better file handling
    file_path = os.path.join(os.getcwd(), "data", "product_reviews.csv")

    # Initialize data preprocessor
    preprocessor = DataPreprocessor()

    # Initialize embedding generator
    embedder = EmbeddingGenerator()

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df["review_text_cleaned"] = df["review_text"].apply(preprocessor.clean_text)
        print(df[["review_text", "review_text_cleaned"]].head())

        # Generate embeddings for each review
        df["embedding"] = df["review_text_cleaned"].apply(embedder.generate_embedding)
        print(df[["review_text_cleaned", "embedding"]].head())

        # Store reviews in ChromaDB
        store_embeddings_in_chromadb(df)

        # Query similar reviews
        query_text = "Amazing battery life and great sound quality"
        results = retrieve_similar_reviews(query_text, top_n=3)

        for i, res in enumerate(results):
            print(f"Rank {i+1}: {res['review_text']} (Product: {res['product']}, Rating: {res['rating']})")
    else:
        print(f"Error: File not found at {file_path}")