import os
import json
import requests
import numpy as np

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

def get_stock_data(target, api_key, ohcl='4. close'):
    """ Fetches the stock data of a specified target company to a dictionary.
        data['Dates'] stores the dates (e.g. ['2022-09-06', ...]) and
        data['Prices'] stores corresponding stock prices (e.g. [167.0, ...]).
        They are sorted in chronological order.
        Optional keyword argument ohcl represents open, high, low, or close
        daily prices. """
    if ohcl not in ['1. open', '2. high', '3. low', '4. close']:
        raise TypeError('ohcl argument must be "1. open", "2. high", "3. low", or "4. close"')
    if isinstance(target, str):
        target = grab_daily_data(target, api_key)
    stock_data = { 'Dates': [], 'Prices': [] }
    for day in target['Time Series (Daily)']:
        stock_data['Dates'].insert(0, day)
        stock_data['Prices'].insert(0, target['Time Series (Daily)'][day][ohcl])
    return stock_data

def preprocess_data(target, api_key):
    """ Preprocesses stock data for machine learning.
        Returns three data: x, y, and time_series.
        x variable stores past 365 stock price data points and
        y variable stores average stock prices of the next 5 data points.
        (e.g. x: [[167.0, ...], ...], y: [173.24000, ...])
        In addition, returns time_series representing the date y-value represents. """
    stock_data = get_stock_data(target, api_key)
    time_series = []
    stock_prices = np.array(stock_data['Prices']).astype(float)
    x = []
    y = []
    for i in range(0, len(stock_prices) - 369):
        x.append(stock_prices[0 + i:365+i])
        y.append(sum(stock_prices[365+i:370+i]) / 5)
        time_series.append(stock_data['Dates'][i+365])
    return x, y, time_series

