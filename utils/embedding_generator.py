import json
import os
from typing import Dict, List, Union
from sentence_transformers import SentenceTransformer

class Embeddings:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the embedding model."""
        self.embedding_model = SentenceTransformer(model_name)

    @staticmethod
    def normalize_stock_name(name: str) -> str:
        """Normalize stock names by removing common suffixes."""
        suffixes = ["Inc.", "Ltd.", "Corporation", "Company", "Co."]
        for suffix in suffixes:
            name = name.replace(suffix, "").strip()
        return name

    @staticmethod
    def load_json(file_path: str) -> Union[Dict, List]:
        """Load JSON data from a file."""
        file_path = os.path.join(os.path.dirname(__file__), "..", file_path)
        with open(file_path, "r") as file:
            return json.load(file)

    def generate_weather_embeddings(self, city_data: List[Dict]) -> List[Dict]:
        """Generate embeddings for weather-related data."""
        weather_embeddings = []
        for item in city_data:
            city_name = item["name"]
            embedding = self.embedding_model.encode(city_name)
            weather_embeddings.append({
                "vector": embedding,
                "metadata": {"type": "weather", "name": city_name}
            })
        return weather_embeddings

    def generate_crypto_embeddings(self, crypto_data: Dict[str, str]) -> List[Dict]:
        """Generate embeddings for cryptocurrency-related data."""
        crypto_embeddings = []
        for ticker, name in crypto_data.items():
            ticker_embedding = self.embedding_model.encode(ticker)
            crypto_embeddings.append({
                "vector": ticker_embedding,
                "metadata": {"type": "crypto", "ticker": ticker, "name": name}
            })

            name_embedding = self.embedding_model.encode(name)
            crypto_embeddings.append({
                "vector": name_embedding,
                "metadata": {"type": "crypto", "ticker": ticker, "name": name}
            })
        return crypto_embeddings

    def generate_stock_embeddings(self, stock_data: Dict[str, str]) -> List[Dict]:
        """Generate embeddings for stock-related data."""
        stock_embeddings = []
        for ticker, name in stock_data.items():
            normalized_name = self.normalize_stock_name(name)

            ticker_embedding = self.embedding_model.encode(ticker)
            stock_embeddings.append({
                "vector": ticker_embedding,
                "metadata": {"type": "stock", "ticker": ticker, "name": normalized_name}
            })

            name_embedding = self.embedding_model.encode(normalized_name)
            stock_embeddings.append({
                "vector": name_embedding,
                "metadata": {"type": "stock", "ticker": ticker, "name": normalized_name}
            })
        return stock_embeddings