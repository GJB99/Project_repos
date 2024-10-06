import requests
import json
import datetime
import csv
import pandas as pd
import time

def get_coin_id(name):
    url = f'https://api.coingecko.com/api/v3/coins/list'
    response = requests.get(url)
    if response.status_code == 200:
        data = json.loads(response.text)
        for coin in data:
            if coin['name'].lower() == name.lower():
                return coin['id']
        raise Exception(f'Coin {name} not found.')
    else:
        raise Exception(f'Error: {response.status_code}')

def get_historical_data(coin_id, from_timestamp, to_timestamp):
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range?vs_currency=usd&from={from_timestamp}&to={to_timestamp}'
    response = requests.get(url)

    if response.status_code == 200:
        return json.loads(response.text)
    else:
        raise Exception(f'Error: {response.status_code}')

def save_data_to_csv(data, file_path):
    with open(file_path, 'w', newline='') as csvfile:
        fieldnames = ['timestamp', 'price']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(len(data['prices'])):
            writer.writerow({'timestamp': data['prices'][i][0], 'price': data['prices'][i][1]})

def read_csv(file_path):
    return pd.read_csv(file_path)

def main():
    # List of cryptocurrency names
    crypto_names = ['BlackCoin']
    # Set the date range for historical data
    from_date = datetime.datetime(2013, 4, 6)  # Starting 10 years ago
    to_date = datetime.datetime(2023, 4, 6)  # Up to now

    # Convert dates to Unix timestamps
    from_timestamp = int(from_date.timestamp())
    to_timestamp = int(to_date.timestamp())

    # Fetch the list of coins only once to avoid unnecessary requests
    coin_list_url = 'https://api.coingecko.com/api/v3/coins/list'
    coin_list_response = requests.get(coin_list_url)
    coin_list = json.loads(coin_list_response.text)

    for name in crypto_names:
        try:
            coin_id = get_coin_id(name)
            historical_data = get_historical_data(coin_id, from_timestamp, to_timestamp)

            # Save the data to a CSV file on your desktop
            csv_file_path = f'{name}_data.csv'  # Replace 'your_desktop_path' with the path to your desktop
            save_data_to_csv(historical_data, csv_file_path)

            # Read the saved CSV file using pandas
            df = read_csv(csv_file_path)
            print(f'{name} historical data:')
            print(df)
            print('----------------------------------------------------')

            # Introduce a delay between requests to avoid rate limits
            time.sleep(50)
        except Exception as e:
            print(f'Error fetching data for {name}: {e}')
            print('----------------------------------------------------')

if __name__ == '__main__':
    main()