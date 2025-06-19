import json
import csv


with open('./stock.json') as f:
    stock = json.load(f)
with open('./crypto.json') as f:
    crypto = json.load(f)
with open('./city_list.json') as f:
    cities = json.load(f)

def run(file_name):
    if file_name == "city":
         with open(f'./{file_name}_mapppings.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['text', 'label'])
            
            for city in cities:
                if isinstance(city, dict) and 'name' in city:
                    name = city['name'].strip()
                    writer.writerow([name, name])

    elif file_name == "stock":
        with open(f'./{file_name}_mapppings.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['text', 'label'])
            for name, symbol in stock.items():
                writer.writerow([name, symbol])

    elif file_name == "crypto":
        with open(f'./{file_name}_mapppings.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['text', 'label'])
            for name, symbol in crypto.items():
                writer.writerow([name, symbol])


run("city")