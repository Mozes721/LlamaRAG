from utils.helper import Helper
import requests
from config.apis import APIkeys, APIEndpoints

class APIRequests:
    def get_crypto_data(self, crypto):
        api = APIEndpoints()
        api.CRYPTO_PARAMETERS['slug'] = crypto
        api.CRYPTO_HEADERS['X-CMC_PRO_API_KEY'] = APIkeys.cryptoAPI

        session = requests.Session()
        session.headers.update(APIEndpoints.CRYPTO_HEADERS)

        response = session.get(APIEndpoints.CRYPTO_URL,
                               params=APIEndpoints.CRYPTO_PARAMETERS)
        response_json = response.json()
        (crypto_name,
         crypto_symbol,
         formatted_price,
         formatted_market_cap) = Helper.extract_crypto_info(response_json)

        return {
            "name": crypto_name,
            "symbol": crypto_symbol,
            "price": formatted_price,
            "market_cap": formatted_market_cap
        }

    def get_stock_data(self, ticker):
        url = APIEndpoints.STOCK_URL.format(symbol=ticker,
                                         api_key=APIkeys.stockAPI)
        json_data = requests.get(url).json()
        get_values = json_data['values']
        get_open_position = get_values[0]['open']

        return {
            "ticker": ticker,
            "price": float(get_open_position)
        }

    def get_weather_data(self, location):
        city_id = Helper.get_city_id(location)
        url = APIEndpoints.WEATHER_URL.format(api_key=APIkeys.weatherAPI,
                                           location=city_id)
        weather_json = requests.get(url).json()
        weather_data = Helper.extract_weather_info(weather_json)

        return {
            "location": location,
            "temp": weather_data["temp"],
            "weather": weather_data["weather"],
            "wind_speed": weather_data["wind_speed"]
        }