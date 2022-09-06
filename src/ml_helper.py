import json
import os
import matplotlib as plt
import pandas as pd
import numpy as np
import tensorflow

def predict_and_save(symbol, api_key, model):
    x, y, time_series = preprocess_data(symbol, api_key)
    pred = model.predict(np.asarray(x))
    stock_data = get_stock_data(symbol, api_key)
    plt.plot(pd.to_datetime(stock_data['Dates'], np.array(stock_data['Prices']).astype(float)))
    plt.plot(pd.to_datetime(time_series), pred)
    plt.legend(['Actual Price', 'Predicted Price'])
    plt.show()
    my_dict = {str(time_series[i]): str(pred[i][0]) for i in range(0, len(pred))}
    with open(os.path.join('..', 'data', symbol + '_pred.json'), 'w') as output_file:
        json.dump(my_dict, output_file)

