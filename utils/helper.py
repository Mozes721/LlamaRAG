import json
import os
from typing import Dict, List, Union

class Helper:
    @staticmethod
    def load_json(file_path: str) -> Union[Dict, List]:
        """Load JSON data from a file."""
        file_path = os.path.join(os.path.dirname(__file__), "..", file_path)
        with open(file_path, "r") as file:
            return json.load(file)
    
    @staticmethod
    def track_progress(current: int, total: int, task_name: str):
        progress = (current / total) * 100
        print(f"\rProcessing {task_name}: {progress:.2f}% complete", end="")
        if current == total:
            print()
        
    @staticmethod
    def normalize_stock_name(name: str) -> str:
        """Normalize stock names by removing common suffixes."""
        suffixes = ["Inc.", "Ltd.", "Corporation", "Company", "Co."]
        for suffix in suffixes:
            name = name.replace(suffix, "").strip()
        return name
    
    @staticmethod
    def extract_crypto_info(crypto_data):
        """Extract and format cryptocurrency information from API response."""
        crypto_name = crypto_data['data']['1']['name']
        crypto_symbol = crypto_data['data']['1']['symbol']
        crypto_price = crypto_data['data']['1']['quote']['USD']['price']
        crypto_market_cap = crypto_data['data']['1']['quote']['USD']['market_cap']

        formatted_price = "{:.2f}".format(crypto_price)
        formatted_market_cap = "{:.2f}".format(crypto_market_cap)

        return (
            crypto_name,
            crypto_symbol,
            formatted_price,
            formatted_market_cap
        )

    @staticmethod
    def get_city_id(city):
        json_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'city_list.json')
        
        # Check if the file exists
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"city_list.json not found at: {json_file_path}")
        
        # Load and search for the city
        with open(json_file_path) as json_file:
            data = json.load(json_file)
            for item in data:
                if item['name'] == city:
                    return item['id']
        return None

    @staticmethod
    def extract_weather_info(weather_data):
        """Extract and format weather information from API response."""
        weather_info = {}

        weather_info['weather'] = weather_data['weather'][0]['main']
        temp = str(weather_data['main']['temp'])
        weather_info['temp'] = temp[:2] + '.' + temp[2:3]
        weather_info['wind_speed'] = weather_data['wind']['speed']

        return weather_info