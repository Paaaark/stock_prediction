import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from data_manager import grab_daily_data

def plot_swing_chart(symbol, api_key, n=50, upper_color='g', lower_color='r'):
    df = grab_daily_data(symbol, api_key)
    df = pd.DataFrame.from_dict(df['Time Series (Daily)'], orient='index')
    df = df.sort_index().astype(float).tail(n=n)
    df['Mean 1'] = (df['4. close'] + df['1. open']) / 2
    df['y err 1'] = abs(df['Mean 1'] - df['1. open'])
    df['Mean 2'] = (df['3. low'] + df['2. high']) / 2
    df['y err 2'] = df['2. high'] - df['Mean 2']
    df['color'] = np.where(df['4. close'] > df['1. open'], upper_color, lower_color)

    plt.errorbar(x=pd.to_datetime(df.index), y=df['Mean 2'], yerr=df['y err 2'],
                 fmt='none', ecolor=df['color'])
    plt.errorbar(x=pd.to_datetime(df.index), y=df['Mean 1'], yerr=df['y err 1'],
                 fmt='none', ecolor=df['color'], elinewidth=3.5)
    plt.show()
