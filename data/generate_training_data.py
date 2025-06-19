import json
import random

def load_json_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def generate_stock_queries(stock_data):
    queries = []
    templates = [
        "What is the current price of {name} stock?",
        "How much is {ticker} trading at?",
        "What's the current price of {name}?",
        "Can you tell me the current price of {ticker}?",
        "I'd like to know the current price of {name} stock",
        "What is {ticker} trading at right now?",
        "How much is {name} stock worth?",
        "Could you check the current price of {ticker}?",
        "What's the current value of {name} stock?",
        "Tell me the current price of {ticker}"
    ]
    
    for ticker, name in stock_data.items():
        for template in templates:
            query = template.format(ticker=ticker, name=name)
            queries.append({
                "instruction": query,
                "input": "",
                "output": f"The current price of {name} stock is {{price}} USD"
            })
    return queries

def generate_crypto_queries(crypto_data):
    queries = []
    templates = [
        "What is the current price of {name}?",
        "How much is {ticker} worth?",
        "What's the current price of {ticker}?",
        "Can you tell me the current price of {name}?",
        "I'd like to know the current price of {ticker}",
        "What is {name} trading at right now?",
        "How much is {ticker} trading for?",
        "Could you check the current price of {name}?",
        "What's the current value of {ticker}?",
        "Tell me the current price of {name}"
    ]
    
    for ticker, name in crypto_data.items():
        for template in templates:
            query = template.format(ticker=ticker, name=name)
            queries.append({
                "instruction": query,
                "input": "",
                "output": f"The price of {name} is {{price}} USD"
            })
    return queries

def generate_weather_queries():
    cities = [
        "New York", "London", "Tokyo", "Paris", "Sydney",
        "Berlin", "Rome", "Madrid", "Amsterdam", "Singapore",
        "Dubai", "Hong Kong", "Seoul", "Mumbai", "Toronto",
        "Los Angeles", "Chicago", "Houston", "Miami", "San Francisco"
    ]
    
    queries = []
    templates = [
        "What is the weather in {city}?",
        "How's the weather in {city} today?",
        "What's the current weather in {city}?",
        "Can you tell me the weather in {city}?",
        "I'd like to know the weather in {city}",
        "What's the temperature in {city}?",
        "How's the weather looking in {city}?",
        "Could you check the weather in {city}?",
        "What's the current temperature in {city}?",
        "Tell me the weather in {city}"
    ]
    
    for city in cities:
        for template in templates:
            query = template.format(city=city)
            queries.append({
                "instruction": query,
                "input": "",
                "output": f"Current weather in {city}: temperature at {{temp}}°C, the sky is {{weather}} and wind speed ranges at {{wind_speed}} m/s"
            })
    return queries

def main():
    # Load data
    stock_data = load_json_file('stock.json')
    crypto_data = load_json_file('crypto.json')
    
    # Generate queries
    stock_queries = generate_stock_queries(stock_data)
    crypto_queries = generate_crypto_queries(crypto_data)
    weather_queries = generate_weather_queries()
    
    # Combine all queries
    all_queries = stock_queries + crypto_queries + weather_queries
    
    # Shuffle the queries
    random.shuffle(all_queries)
    
    # Create the final dataset
    dataset = {
        "version": "1.0",
        "data": all_queries
    }
    
    # Save to file
    with open('fine_tuning_data.json', 'w') as f:
        json.dump(dataset, f, indent=4)
    
    print(f"Generated {len(all_queries)} training examples:")
    print(f"- {len(stock_queries)} stock queries")
    print(f"- {len(crypto_queries)} crypto queries")
    print(f"- {len(weather_queries)} weather queries")

if __name__ == "__main__":
    main() 