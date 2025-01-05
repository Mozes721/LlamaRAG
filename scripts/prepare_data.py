from config.apis import APIkeys
from typing import Dict, List
from upstash_vector import Index
from utils.embedding_generator import Embeddings
import os

def generate_embeddings():
    """Generate and store embeddings for weather, crypto, and stock data."""
    # Initialize Upstash Vector client
    index = Index(url=APIkeys.upstashAPI, token=APIkeys.upstashToken)

    # Initialize Embeddings class
    embedding_generator = Embeddings()


    # Load data using os to construct paths
    city_data = embedding_generator.load_json(os.path.join("data", "city_list.json"))
    crypto_data = embedding_generator.load_json(os.path.join("data", "crypto_list.json"))
    stock_data = embedding_generator.load_json(os.path.join("data", "stock_list.json"))

    print("Data loaded successfully.")
    # Generate embeddings
    weather_embeddings = embedding_generator.generate_weather_embeddings(city_data)
    crypto_embeddings = embedding_generator.generate_crypto_embeddings(crypto_data)
    stock_embeddings = embedding_generator.generate_stock_embeddings(stock_data)

    # Store embeddings in Upstash Vector DB
    def store_embeddings(embeddings: List[Dict]):
        for embedding in embeddings:
            index.upsert(
                vectors=[
                    (embedding["metadata"]["name"], embedding["vector"], embedding["metadata"])
                ]
            )

    # Store all embeddings
    store_embeddings(weather_embeddings)
    store_embeddings(crypto_embeddings)
    store_embeddings(stock_embeddings)

    print("Embeddings stored in Upstash Vector DB.")