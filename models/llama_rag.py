from config.apis import APIkeys
from typing import Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from upstash_vector import Index
from sentence_transformers import SentenceTransformer
from api.requests import APIRequests  # Directly import APIRequests


class LlamaRAG:
    def __init__(self):
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize Upstash Vector client
        self.index = Index(url=APIkeys.upstashAPI, token=APIkeys.upstashToken)

        # Load Llama tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf", 
            torch_dtype=torch.float16, 
            device_map="auto"
        )

        # Move model to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        # Initialize API requests directly
        self.api_requests = APIRequests()  # No need for APIService

    def generate_response(self, prompt: str) -> str:
        """Generate a response using the Llama model."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(inputs.input_ids, max_length=200)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def retrieve_data(self, query: str, top_k: int = 1) -> Optional[Dict]:
        """Retrieve relevant data from Upstash Vector DB."""
        query_embedding = self.embedding_model.encode(query)
        results = self.index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        return results[0].metadata if results else None

    def handle_query(self, query: str) -> str:
        """Handle user queries and generate a dynamic response using Llama."""
        # Retrieve relevant data
        metadata = self.retrieve_data(query)
        
        if not metadata:
            # If no metadata is found, ask Llama to generate a response
            prompt = (
                f"The user asked: '{query}'. "
                "However, I couldn't find any relevant information in my database. "
                "Please provide a helpful and polite response to the user."
            )
            return self.generate_response(prompt)
        
        # Determine the type of query from metadata
        query_type = metadata.get("type")
        
        if query_type == "weather":
            city_name = metadata.get("name")
            if not city_name:
                return "Sorry, I couldn't find the city in the query."
            # Fetch raw weather data directly
            weather_data = self.api_requests.get_weather_data(city_name)
            # Generate dynamic response using Llama
            prompt = (
                f"Provide a detailed weather report for {city_name} based on the following data: "
                f"Temperature: {weather_data['temp']}°C, "
                f"Weather: {weather_data['weather']}, "
                f"Wind Speed: {weather_data['wind_speed']} km/h. "
                "Make the response sound natural and engaging."
            )
            return self.generate_response(prompt)
        
        elif query_type == "crypto":
            crypto_name = metadata.get("name")
            crypto_ticker = metadata.get("ticker")
            if not crypto_name or not crypto_ticker:
                return "Sorry, I couldn't find the crypto in the query."
            # Fetch raw crypto data directly
            crypto_data = self.api_requests.get_crypto_data(crypto_ticker)
            # Generate dynamic response using Llama
            prompt = (
                f"Provide a detailed report on the current price and market trends of {crypto_name} ({crypto_ticker}) "
                f"based on the following data: Price: {crypto_data['price']}, "
                f"Market Cap: {crypto_data['market_cap']}. "
                "Make the response sound natural and engaging."
            )
            return self.generate_response(prompt)
        
        elif query_type == "stock":
            stock_name = metadata.get("name")
            stock_ticker = metadata.get("ticker")
            if not stock_name or not stock_ticker:
                return "Sorry, I couldn't find the stock in the query."
            # Fetch raw stock data directly
            stock_data = self.api_requests.get_stock_data(stock_ticker)
            # Generate dynamic response using Llama
            prompt = (
                f"Provide a detailed report on the current price and market trends of {stock_name} ({stock_ticker}) "
                f"based on the following data: Price: {stock_data['price']} USD. "
                "Make the response sound natural and engaging."
            )
            return self.generate_response(prompt)
        
        else:
            return "Please ask a concrete question about weather, stock prices, or crypto prices."