import os
import json
import requests
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def grab_daily_data(company_label, api_key):
    """ Grabs daily stock data of a specificed company_label.
        If the data exsits, fetch it from the json file.
        If not, fetch it from alpha vantage api and save it as a json file.
        Returns the data in dictionary loaded from json file. """
    file_path = os.path.join('..', 'data', company_label + '.json')
    if os.path.exists(file_path):
        f = open(file_path)
        return json.load(f)
    url = 'https://www.alphavantage.co/' + 'query?function=TIME_SERIES_DAILY&symbol=' + company_label + '&outputsize=full&apikey=' + api_key
    r = requests.get(url)
    data = r.json()
    with open(os.path.join('..', 'data', company_label + '.json'), "w") as output_file:
        json.dump(data, output_file)
    return data;

def get_one_to_one_stock_data(target, api_key, ohcl='4. close'):
    if ohcl not in ['1. open', '2. high', '3. low', '4. close']:
        raise TypeError('ohcl argument must be "1. open", "2. high", "3. low" or "4. close"')
    if isinstance(target, str):
        target = grab_daily_data(target, api_key)
    stock_data = {}
    for day in target['Time Series (Daily)']:
        stock_data[day] = float(target['Time Series (Daily)'][day][ohcl])
    return stock_data

def get_stock_data(target, api_key, ohcl='4. close'):
    """ Fetches the stock data of a specified target company to a dictionary.
        data['Dates'] stores the dates (e.g. ['2022-09-06', ...]) and
        data['Prices'] stores corresponding stock prices (e.g. [167.0, ...]).
        They are sorted in chronological order.
        Optional keyword argument ohcl represents open, high, low, or close
        daily prices. """
    if ohcl not in ['1. open', '2. high', '3. low', '4. close', '5. all']:
        raise TypeError('ohcl argument must be "1. open", "2. high", "3. low", or "4. close", or "5. all')
    if isinstance(target, str):
        target = grab_daily_data(target, api_key)
    stock_data = { 'Dates': [], 'Prices': [] }
    for day in target['Time Series (Daily)']:
        stock_data['Dates'].insert(0, day)
        stock_data['Prices'].insert(0, target['Time Series (Daily)'][day][ohcl])
    stock_data['Prices'] = np.array(stock_data['Prices']).astype(float)
    return stock_data

def preprocess_data(target, api_key, start=None, end=None):
    """ Preprocesses stock data at the specified start and end index.
        Returns three data: x, y, and time_series.
        x variable stores past 365 stock price data points and
        y variable stores average stock prices of the next 5 data points.
        (e.g. x: [[167.0, ...], ...], y: [173.24000, ...])
        In addition, returns time_series representing the date y-value represents. """
    if (start is None and end is not None) or (start is not None and end is None):
        raise ValueError('start and end has to be both non-None or both None')
    stock_data = get_stock_data(target, api_key)
    time_series = []
    stock_prices = np.array(stock_data['Prices']).astype(float)
    if start is not None and end is not None:
        stock_prices = stock_prices[start:end]
    x = []
    y = []
    for i in range(0, len(stock_prices) - 369):
        x.append(stock_prices[0 + i:365+i])
        # y.append(stock_prices[365+i])
        y.append(sum(stock_prices[365+i:370+i]) / 5)
        time_series.append(stock_data['Dates'][i+365])
    return x, y, time_series

def preprocess_data_cnn_1d(target, api_key):
    x, y, time_series = preprocess_data(target, api_key)
    x = np.reshape(x, (len(x), 365, 1))
    return x, y, time_series

def preprocess_data_rnn(target, api_key, lookback=10):
    stock_data = grab_daily_data(target, api_key)
    df = pd.DataFrame.from_dict(stock_data['Time Series (Daily)'], orient='index')
    df = df.drop(labels='5. volume', axis=1)
    df = df.sort_index()
    df['Target'] = df['4. close'].astype(float).shift(-1)
    df = df.dropna()

    scaler = MinMaxScaler()
    scaler = scaler.fit(df)
    arr = scaler.transform(df)
    df = pd.DataFrame(arr, columns=['open', 'high', 'low', 'close', 'Target'])

    x_arr = df.drop(labels='Target', axis=1).to_numpy().astype(float)
    y_arr = df['Target'].to_numpy().astype(float)
    x = []
    y = []
    for i in range(lookback, len(x_arr) + 1):
        x.append(x_arr[i-lookback: i])
        y.append(y_arr[i - 1])
    y = np.reshape(y, (len(y), 1))
    return x, y


def preprocess_data_multiple(symbols, api_key):
    x = []
    y = []
    for symbol in symbols:
        curr_x, curr_y, time_series = preprocess_data(symbol, api_key)
        x.extend(curr_x)
        y.extend(curr_y)
    return x, y

