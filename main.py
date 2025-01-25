import transformers
import torch
import requests

def main():
    # Check if CUDA (GPU) is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        # Load a zero-shot classification model
        print("Loading zero-shot classification model...")
        zero_shot_pipeline = transformers.pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",  # A model suitable for zero-shot classification
            device_map="auto" if device == "cuda" else None,
        )
        print("Zero-shot classification model loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # Start interactive loop
    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        try:
            # Step 1: Classify the query using zero-shot classification
            print("Classifying query...")
            categories = ["crypto", "stock", "weather"]
            classification_result = zero_shot_pipeline(query, candidate_labels=categories)
            category = classification_result["labels"][0]  # Get the top predicted category
            print(f"Query classified as: {category}")

            # Step 2: Fetch data based on the category
            if category == "crypto":
                # Extract the cryptocurrency name (e.g., "bitcoin" from "What is the price of bitcoin?")
                crypto_name = query.lower().split()[-1]  # Simple extraction (improve as needed)
                response = fetch_crypto_price(crypto_name)
            elif category == "weather":
                # Extract the city name (e.g., "Seattle" from "What is the weather in Seattle?")
                city_name = query.lower().split()[-1]  # Simple extraction (improve as needed)
                response = fetch_weather_data(city_name)
            elif category == "stock":
                # Extract the stock ticker (e.g., "AAPL" from "What is the stock price of AAPL?")
                stock_ticker = query.lower().split()[-1]  # Simple extraction (improve as needed)
                response = fetch_stock_price(stock_ticker)
            else:
                response = "Sorry, I couldn't understand your query."

            # Step 3: Display the response
            print(f"Bot: {response}")
        except Exception as e:
            print(f"Error handling query: {e}")

def fetch_crypto_price(crypto_name: str) -> str:
    """
    Fetch the price of a cryptocurrency using an API.
    """
    # Mock API call (replace with actual API)
    crypto_prices = {
        "bitcoin": 30000,
        "ethereum": 2000,
    }
    price = crypto_prices.get(crypto_name, "Unknown cryptocurrency")
    return f"The price of {crypto_name} is ${price} USD."

def fetch_weather_data(city_name: str) -> str:
    """
    Fetch weather data for a city using an API.
    """
    # Mock API call (replace with actual API)
    weather_data = {
        "seattle": {"temperature": 15, "condition": "Rainy"},
        "new york": {"temperature": 25, "condition": "Sunny"},
    }
    data = weather_data.get(city_name, "Unknown city")
    if isinstance(data, dict):
        return f"The weather in {city_name} is {data['temperature']}°C and {data['condition']}."
    else:
        return data

def fetch_stock_price(stock_ticker: str) -> str:
    """
    Fetch the price of a stock using an API.
    """
    # Mock API call (replace with actual API)
    stock_prices = {
        "aapl": 150,
        "googl": 2800,
    }
    price = stock_prices.get(stock_ticker, "Unknown stock")
    return f"The stock price of {stock_ticker.upper()} is ${price} USD."

if __name__ == "__main__":
    main()