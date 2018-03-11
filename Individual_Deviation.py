
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import matplotlib.dates as mdates
import numpy as np
import math


from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)

path1=os.getcwd()
path0=os.path.dirname(path1)
path2=path0+'/Output'



def getSTD(currency):
    DFrame=pd.read_csv("coin_data.csv")
    DFrame['mean']=DFrame[['open','close']].mean(axis=1)


    for region, df_Type in DFrame.groupby('currency'):
        if str(region).lower() == currency:
            df_Type['7Roll'] = df_Type['mean'].rolling(7, min_periods=5).mean()
            df_Type['7MSE'] = (df_Type['mean'] - df_Type['7Roll']) ** 2
            df_Type['7Var'] = df_Type['7MSE'].rolling(7, min_periods=5).mean()
            df_Type['7STD'] = (df_Type['7Var']) ** 0.5

            return df_Type['7STD']


    #     num+=1
    #     title= str(region)
    #     X= [dt.datetime.strptime(d,'%d/%m/%Y').date() for d in df_Type['date']]
    #     fig,ax=plt.subplots()
    #     plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    #     plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    #     line1, = ax.plot(X, '7STD', data=df_Type,color='g',linewidth=0.4)
    #     line2, = ax.plot(X, '30STD',  data=df_Type,color='b',linewidth=0.4)
    #     line3, = ax.plot(X, '365STD', data=df_Type,color='r',linewidth=0.4)
    #
    #     ax.set_title(r'Standard dev for coin, %s' %(title))
    #     ax.set_xlabel(r'Date')
    #     ax.set_ylabel(r'Price', rotation='vertical')
    #     if len(X)<20:
    #         ax.set_xticks(X)
    #     else:
    #         ax.set_xticks((X[::math.floor(len(X)/20)]))
    #     plt.gcf().autofmt_xdate
    #     plt.xticks(rotation=70)
    #     plt.legend([line1,line2,line3],['Weekly','Monthly','Yearly'])
    #     plt.tight_layout()
    #
    #     plt.savefig(path2+'/Rolling_Vol/'+title+'.pdf')
    #     plt.close()
    #
    #     fig, ax=plt.subplots()
    #     plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    #     plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    #     line1, = ax.plot(X, '7Var', data=df_Type,color='g',linewidth=0.4)
    #     line2, = ax.plot(X, '30Var',  data=df_Type,color='b',linewidth=0.4)
    #     line3, = ax.plot(X, '365Var', data=df_Type,color='r',linewidth=0.4)
    #
    #     ax.set_title(r'Variance for coin, %s' %(title))
    #     ax.set_xlabel(r'Date')
    #     ax.set_ylabel(r'Price', rotation='vertical')
    #     if len(X)<20:
    #         ax.set_xticks(X)
    #     else:
    #         ax.set_xticks((X[::math.floor(len(X)/20)]))
    #     plt.gcf().autofmt_xdate
    #     plt.xticks(rotation=70)
    #     plt.legend([line1,line2,line3],['Weekly','Monthly','Yearly'])
    #     plt.tight_layout()
    #
    #     plt.savefig(path2+'/Rolling_Var/'+title+'.pdf')
    #     plt.close()
    #
    # print(num)
