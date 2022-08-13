import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

def load_ts(fname):
    """return df of timeseries, fname is a csv file, assume the dates are sorted"""
    def str_to_datetime(s):
        split = s.split('/')
        month, day, year = int(split[0]), int(split[1]), int(split[2])
        return datetime.datetime(year=year, month=month, day=day)

    def price_to_float(s):
        return float(s[1:])

    df = pd.read_csv(fname)
    df = df[['Date', 'Close/Last', 'Volume']]
    df = df.rename(columns={'Close/Last':'Close'})
    df['Date'] = df['Date'].apply(str_to_datetime)
    df['Close'] = df['Close'].apply(price_to_float)
    df.index = df.pop('Date')

    if df.isnull().values.any():
        print(f"{fname} has NaN values, dropping rows with nan values")
        df = df.dropna()

    return df.sort_index()

def load_all():
    """return list of names and df's,
    will worry bout memory if it becomes a problem"""
    names = []
    dfs = []
    for fname in glob.glob("data/*.csv"):
        names.append(os.path.basename(fname)[:-4])
        dfs.append(load_ts(fname))
    return names, dfs

def load_all_test():
    names = []
    dfs = []
    for fname in glob.glob("data_val/*.csv"):
        names.append(os.path.basename(fname)[:-4])
        dfs.append(load_ts(fname))
    return names, dfs

def normalize0(df, margin=.25):
    """ 
    Normalize from 0 to 1.25 historical max
    df w/ columns [date, close, volume]
    Margin within [0,1]
    return df_norm, factor1, factor2
    """
    factor1 = df['Close'].max() * (1+margin)
    df['Close'] /= factor1

    factor2 = df['Volume'].max() * (1+margin)
    df['Volume'] /= factor2

    return factor1, factor2

# def normalize1(df, margin=.25):
#     """
#     complex normalization aiming to amplify price characteristic near 1 
#     works as intended but abandoned because price exceeds 1 far too easily!
#     terrible for real-time application
#     """
#     def logit(p):
#         return np.log(p) - np.log(1 - p)
#     normalize0(df, margin)
#     df['Close'] = logit(df['Close'] / 2 + 0.5)
#     normalize0(df, margin)

def randepisode(df, length):
    start = np.random.randint(0, df.shape[0]+1-length)
    end = start + length
    return df.iloc[start:end].copy()

if __name__ == "__main__":
    load_all()
    df = load_ts("data/ADBE.csv")

    plt.figure()
    df = randepisode(df, 100)
    normalize0(df)
    df.plot()
    plt.show()



    # plt.figure()
    # #plt.plot(df.index, df['Close'])
    # df = randepisode(df, 100)
    # normalize1(df)
    # df.plot()
    # #plt.plot(df.index, df)
    # plt.show()
    # abde = pd.read_csv("data/source/ADBE.csv")#, index_col="Date")
    # df = abde[['Date', 'Close/Last']]
    # df = df.rename(columns={'Close/Last':'Close'})
    # df['Date'] = df['Date'].apply(str_to_datetime)
    # df['Close'] = df['Close'].apply(price_to_float)
    # df.index = df.pop('Date')

    # plt.plot(df.index, df['Close'])

