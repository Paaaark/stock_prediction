import os
import json
import requests

def grab_daily_data(company_label, api_key):
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
