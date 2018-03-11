import pandas as pd


def getSTD(currency):
    # gets the STD of a currency over 7 days

    DFrame = pd.read_csv("coin_data.csv")
    DFrame['mean']=DFrame[['open','close']].mean(axis=1)


    for region, df_Type in DFrame.groupby('currency'):
        if str(region).lower() == currency:
            df_Type['7Roll'] = df_Type['mean'].rolling(7, min_periods=5).mean()
            df_Type['7MSE'] = (df_Type['mean'] - df_Type['7Roll']) ** 2
            df_Type['7Var'] = df_Type['7MSE'].rolling(7, min_periods=5).mean()
            df_Type['7STD'] = (df_Type['7Var']) ** 0.5

            return df_Type['7STD']