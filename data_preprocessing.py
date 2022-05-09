import pandas as pd
import numpy as np

seq_len = 128

def dataclean_preprocessing(data_address):

    df = pd.read_csv(data_address, delimiter=',', usecols=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])

    df['Volume'].replace(to_replace=0, method='ffill', inplace=True)
    df.sort_values('Date', inplace=True)

    # Apply moving average with a window of 10 days to all columns
    df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].rolling(10).mean()
    df.dropna(how='any', axis=0, inplace=True)

    df['Open'] = df['Open'].pct_change()
    df['High'] = df['High'].pct_change()
    df['Low'] = df['Low'].pct_change()
    df['Close'] = df['Close'].pct_change()
    df['Volume'] = df['Volume'].pct_change()

    times = sorted(df.index.values)
    last_10pct = sorted(df.index.values)[-int(0.1*len(times))] # Last 10% of series
    last_20pct = sorted(df.index.values)[-int(0.2*len(times))] # Last 20% of series

    min_return = min(df[(df.index < last_20pct)][['Open', 'High', 'Low', 'Close']].min(axis=0))
    max_return = max(df[(df.index < last_20pct)][['Open', 'High', 'Low', 'Close']].max(axis=0))

    # Min-max normalize price columns (0-1 range)
    df['Open'] = (df['Open'] - min_return) / (max_return - min_return)
    df['High'] = (df['High'] - min_return) / (max_return - min_return)
    df['Low'] = (df['Low'] - min_return) / (max_return - min_return)
    df['Close'] = (df['Close'] - min_return) / (max_return - min_return)

    min_volume = df[(df.index < last_20pct)]['Volume'].min(axis=0)
    max_volume = df[(df.index < last_20pct)]['Volume'].max(axis=0)

    df['Volume'] = (df['Volume'] - min_volume) / (max_volume - min_volume)

    df_train = df[(df.index < last_20pct)]  # Training data are 80% of total data
    df_val = df[(df.index >= last_20pct) & (df.index < last_10pct)]
    df_test = df[(df.index >= last_10pct)]

    df_train.drop(columns=['Date'], inplace=True)
    df_val.drop(columns=['Date'], inplace=True)
    df_test.drop(columns=['Date'], inplace=True)

    train_data = df_train.values
    val_data = df_val.values
    test_data = df_test.values

    print('Training data shape: {}'.format(train_data.shape))
    print('Validation data shape: {}'.format(val_data.shape))
    print('Test data shape: {}'.format(test_data.shape))

    return train_data, val_data, test_data


def preparing_traindataset(train_data, val_data, test_data):

    X_train, y_train = [], []
    for i in range(seq_len, len(train_data)):
        X_train.append(train_data[i - seq_len:i])
        y_train.append(train_data[:, 3][i])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Validation data
    X_val, y_val = [], []
    for i in range(seq_len, len(val_data)):
        X_val.append(val_data[i - seq_len:i])
        y_val.append(val_data[:, 3][i])
    X_val, y_val = np.array(X_val), np.array(y_val)

    # Test data
    X_test, y_test = [], []
    for i in range(seq_len, len(test_data)):
        X_test.append(test_data[i - seq_len:i])
        y_test.append(test_data[:, 3][i])

    X_test, y_test = np.array(X_test), np.array(y_test)

    print('Training set shape', X_train.shape, y_train.shape)
    print('Validation set shape', X_val.shape, y_val.shape)
    print('Testing set shape', X_test.shape, y_test.shape)

    return X_train, y_train, X_val, y_val, X_test, y_test
