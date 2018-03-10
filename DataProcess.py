#!/usr/bin/env python3

import pandas as pd
import os
import seaborn as sns

path1=os.getcwd()
path0=os.path.dirname(path1)

def getCurrency(currency):
    DFrame=pd.read_csv("coin_data.csv")
    num=0

    for region, df_Type in DFrame.groupby('currency'):
        num+=1
        if str(region).lower()==currency:
            return df_Type['close']
