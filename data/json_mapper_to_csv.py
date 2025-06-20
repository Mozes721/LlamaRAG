import json
import csv


with open('./stock.json') as f:
    stock = json.load(f)
with open('./crypto.json') as f:
    crypto = json.load(f)
with open('./city_list.json') as f:
    cities = json.load(f)

with open('./fine_tuning_data.json') as f:
    data = json.load(f)

def check_if_null_in_fine_tuning_data():
    for i, ex in enumerate(data):
        if ex.get('instruction') is None or ex.get('output') is None:
            print(f"Null at index {i}: {ex}")
        else:
            print(f"Not null at index {i}: {ex}")

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


# run("city")
# check_if_null_in_fine_tuning_data()

def remove_duplicates_from_json():
    # Create a set to track unique examples using instruction and output as key
    seen = set()
    unique_data = []
    duplicates = 0
    
    for item in data:
        # Create a tuple of instruction and output as our uniqueness key
        key = (item['instruction'], item['output'])
        if key not in seen:
            seen.add(key)
            unique_data.append(item)
        else:
            duplicates += 1
    
    print(f"Found {duplicates} duplicates")
    print(f"Original size: {len(data)}")
    print(f"New size: {len(unique_data)}")
    

remove_duplicates_from_json()