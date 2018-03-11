import pandas as pd


def getCurrency(currency):
    # gets all the data from the csv related to a given currency

    DFrame=pd.read_csv("coin_data.csv")
    num=0
    for region, df_Type in DFrame.groupby('currency'):
        num+=1
        if str(region).lower()==currency:
            return df_Type


def allCurrencies():
    # gets a list of all the currencies with more than 50 items in their dataset

    DFrame = pd.read_csv("coin_data.csv")
    names = []

    for region, df_Type in DFrame.groupby('currency'):
        if len(df_Type['close']) >= 50:
            names.append(str(region))

    return names


def getLastDate(currency):
    # gets the date a specific crypto was updated last

    DFrame = pd.read_csv("coin_data.csv")
    num = 0

    for region, df_Type in DFrame.groupby('currency'):
        num += 1
        if str(region).lower() == currency:
            date = df_Type['date'].values[-1]

            return date

