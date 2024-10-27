import pandas as pd
import os



def import_data():
    # Use this function to prepare for the dataset 
    # Output training and test set 
    # Find the data path
    datapath = os.getcwd() + '/Data/'
    # Read all the data
    holiday = pd.read_csv(datapath + 'holidays_events.csv')
    oil = pd.read_csv(datapath + 'oil.csv')
    stores = pd.read_csv(datapath + 'stores.csv')
    train = pd.read_csv(datapath + 'train.csv')
    transactions = pd.read_csv(datapath + 'transactions.csv')
    test = pd.read_csv(datapath + 'test.csv')
    
    # Merge 
    df = pd.merge(holiday,oil,how = 'outer')
    df = pd.merge(df, stores,how = 'outer')
    df = pd.merge(df, train, how = 'outer')
    df = pd.merge(df, transactions, how = 'outer')

    # Convert the date type
    df.date = pd.to_datetime(df.date)
    test.date = pd.to_datetime(test.date)
    
    # Return training set and test set
    return df, test