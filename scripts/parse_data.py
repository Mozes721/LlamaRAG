from pathlib import Path
from config.apis import APIkeys
from typing import Dict, List
from upstash_vector import Index
from utils.helper import Helper
from sentence_transformers import SentenceTransformer

class EmbeddingGenerator:
    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2'):
        self.index = Index(url=APIkeys.upstashAPI, token=APIkeys.upstashToken)
        self.helper = Helper()
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embed_types = ["stock"]
        self.data_files = {
            "weather": "city_list.json",
            "crypto": "crypto_list.json",
            "stock": "stock_list.json"
        }

    def check_embeddings(self) -> List[str]:
        missing_types = []
        for embed_type in self.embed_types:
            try:
                dummy_vector = [0.0] * 1536  # Adjust based on your index's expected dimension
                query_result = self.index.query(vector=dummy_vector, top_k=1)
                print(f"Query result for {embed_type}: {query_result}")  # Debugging line
                matches = query_result  # Adjust based on the actual structure of query_result
                if not any(match.get("metadata", {}).get("type") == embed_type for match in matches):
                    missing_types.append(embed_type)
            except Exception as e:
                print(f"Error checking embeddings for type {embed_type}: {e}")
                missing_types.append(embed_type)
        return missing_types

    def generate_and_store_embeddings(self, types_to_generate: List[str]):
        for embed_type in types_to_generate:
            try:
                data_file = self.data_files.get(embed_type)
                if not data_file:
                    print(f"No data file found for type: {embed_type}")
                    continue

                data = self.helper.load_json(Path("data", data_file))
                generate_method = getattr(self, f"generate_{embed_type}_embeddings")
                embeddings = generate_method(data)
                self.store_embeddings(embeddings, embed_type.capitalize())
                print(f"{embed_type.capitalize()} embeddings generated and stored.")
            except Exception as e:
                raise RuntimeError(f"Error generating {embed_type} embeddings: {e}")

    def generate_weather_embeddings(self, city_data: List[Dict]) -> List[Dict]:
        """Generate embeddings for weather-related data in batches."""
        city_names = [item["name"] for item in city_data]
        total_cities = len(city_names)
        embeddings = self.embedding_model.encode(city_names, batch_size=32)  # Adjust batch_size as needed
        
        weather_embeddings = []
        for i, (city_name, embedding) in enumerate(zip(city_names, embeddings)):
            weather_embeddings.append({
                "vector": embedding,
                "metadata": {"type": "weather", "name": city_name}
            })
            self.helper.track_progress(i + 1, total_cities, "cities")  # Track progress
        return weather_embeddings

    def generate_crypto_embeddings(self, crypto_data: Dict[str, str]) -> List[Dict]:
        """Generate embeddings for cryptocurrency-related data in batches."""
        crypto_items = list(crypto_data.items())
        tickers, names = zip(*crypto_items)
        total_cryptos = len(crypto_items)
        
        # Generate embeddings for tickers and names in batches
        ticker_embeddings = self.embedding_model.encode(tickers, batch_size=32)
        name_embeddings = self.embedding_model.encode(names, batch_size=32)
        
        crypto_embeddings = []
        for i, ((ticker, name), ticker_embedding, name_embedding) in enumerate(zip(crypto_items, ticker_embeddings, name_embeddings)):
            crypto_embeddings.extend([
                {
                    "vector": ticker_embedding,
                    "metadata": {"type": "crypto", "ticker": ticker, "name": name}
                },
                {
                    "vector": name_embedding,
                    "metadata": {"type": "crypto", "ticker": ticker, "name": name}
                }
            ])
            self.helper.track_progress(i + 1, total_cryptos, "cryptos")  # Track progress
        return crypto_embeddings

    def generate_stock_embeddings(self, stock_data: Dict[str, str]) -> List[Dict]:
        """Generate embeddings for stock-related data in batches."""
        stock_items = list(stock_data.items())
        tickers, names = zip(*stock_items)
        normalized_names = [self.helper.normalize_stock_name(name) for name in names]
        total_stocks = len(stock_items)
        
        # Generate embeddings for tickers and normalized names in batches
        ticker_embeddings = self.embedding_model.encode(tickers, batch_size=32)
        name_embeddings = self.embedding_model.encode(normalized_names, batch_size=32)
        
        stock_embeddings = []
        for i, ((ticker, name), ticker_embedding, name_embedding) in enumerate(zip(stock_items, ticker_embeddings, name_embeddings)):
            stock_embeddings.extend([
                {
                    "vector": ticker_embedding,
                    "metadata": {"type": "stock", "ticker": ticker, "name": name}
                },
                {
                    "vector": name_embedding,
                    "metadata": {"type": "stock", "ticker": ticker, "name": name}
                }
            ])
            self.helper.track_progress(i + 1, total_stocks, "stocks")  # Track progress
        return stock_embeddings
    
    def store_embeddings(self, embeddings: List[Dict], embedding_type: str):
        try:
            for embedding in embeddings:
                self.index.upsert(
                    vectors=[
                        (embedding["metadata"]["name"], embedding["vector"], embedding["metadata"])
                    ]
                )
            print(f"{embedding_type} embeddings stored.")
        except Exception as e:
            raise RuntimeError(f"Error storing {embedding_type} embeddings: {e}")