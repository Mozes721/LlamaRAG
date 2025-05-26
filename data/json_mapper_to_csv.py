import json
import csv


with open('./stock.json') as f:
    stock = json.load(f)
# with open('./crypto.json') as f:
#     crypto = json.load(f)
# with open('./city_list.json') as f:
#     cities = json.load(f)

def run():
    with open('./stock_mapper_training.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['text', 'label'])
        
 
        for name, symbol in stock.items():
            writer.writerow([name, symbol])
        # for name, symbol in crypto.items():
        #     writer.writerow([name, symbol])
        # for city in cities:
        #     if isinstance(city, dict) and 'name' in city:
        #         writer.writerow([city['name'].lower(), city['name']])
        #     elif isinstance(city, str):
        #         writer.writerow([city.lower(), city])

run()