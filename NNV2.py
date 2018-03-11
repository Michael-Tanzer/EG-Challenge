import numpy as np
import json
import os
import pandas as pd
import urllib.request
from DataProcessForNN import *
from TFANN import ANNR
import matplotlib.pyplot as mpl


os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def JSONDictToDF(d):
    '''
    Converts a dictionary created from json.loads to a pandas dataframe
    d:      The dictionary
    '''
    n = len(d)
    cols = []
    if n > 0:  # Place the column in sorted order
        cols = sorted(list(d[0].keys()))
    df = pd.DataFrame(columns=cols, index=range(n))
    for i in range(n):
        for coli in cols:
            df.set_value(i, coli, d[i][coli])
    return df


def GetAPIUrl(cur):
    '''
    Makes a URL for querying historical prices of a cyrpto from Poloniex
    cur:    3 letter abbreviation for cryptocurrency (BTC, LTC, etc)
    '''
    u = 'https://poloniex.com/public?command=returnChartData&currencyPair=USDT_' + cur + '&start=1420070400&end=9999999999&period=7200'
    return u


def GetCurDF(cur, fp):
    '''
    cur:    3 letter abbreviation for cryptocurrency (BTC, LTC, etc)
    fp:     File path (to save price data to CSV)
    '''
    openUrl = urllib.request.urlopen(GetAPIUrl(cur))
    r = openUrl.read()
    openUrl.close()
    d = json.loads(r.decode())
    df = JSONDictToDF(d)
    df.to_csv(fp, sep=',')
    return df


# %%Path to store cached currency data
datPath = 'CurDat/'
if not os.path.exists(datPath):
    os.mkdir(datPath)
# Different cryptocurrency types
cl = ['BTC']
# Columns of price data to use
CN = ['close', 'high', 'low', 'open', 'volume']
# Store data frames for each of above types
D = []
for ci in cl:
    dfp = os.path.join(datPath, ci + '.csv')
    try:
        df = pd.read_csv(dfp, sep=',')
    except FileNotFoundError:
        df = GetCurDF(ci, dfp)
    D.append(df)
# %%Only keep range of data that is common to all currency types
cr = min(Di.shape[0] for Di in D)
for i in range(len(cl)):
    D[i] = D[i][(D[i].shape[0] - cr):]


class PastSampler:
    '''
    Forms training samples for predicting future values from past value
    '''

    def __init__(self, N, K):
        '''
        Predict K future sample using N previous samples
        '''
        self.K = K
        self.N = N

    def transform(self, A, Y=None):
        M = self.N + self.K  # Number of samples per row (sample + target)
        # Matrix of sample indices like: {{1, 2..., M}, {2, 3, ..., M + 1}}
        I = np.arange(M) + np.arange(A.shape[0] - M + 1).reshape(-1, 1)
        B = A[I].reshape(-1, M * A.shape[1], *A.shape[2:])
        ci = self.N * A.shape[1]  # Number of features per sample
        return B[:, :ci], B[:, ci:]  # Sample matrix, Target matrix


#%%Features are channels
C = np.hstack((Di[CN] for Di in D))[:, None, :]
HP = 16                 #Holdout period
A = C[0:-HP]
SV = A.mean(axis = 0)   #Scale vector
C /= SV                 #Basic scaling of data

#%%Make samples of temporal sequences of pricing data (channel)
NPS, NFS = 256, 16         #Number of past and future samples
ps = PastSampler(NPS, NFS)
B, Y = ps.transform(A)

NC = B.shape[2]
# 2 1-D conv layers with relu followed by 1-d conv output layer
ns = [('C1d', [8, NC, NC * 2], 4), ('AF', 'relu'),
      ('C1d', [8, NC * 2, NC * 2], 2), ('AF', 'relu'),
      ('C1d', [8, NC * 2, NC], 2)]
# Create the neural network in TensorFlow
cnnr = ANNR(B[0].shape, ns, batchSize=32, learnRate=2e-5,
            maxIter=10, reg=1e-5, tol=1e-2, verbose=True)
cnnr.fit(B, Y)


PTS = []                        #Predicted time sequences
P, YH = B[[-1]], Y[[-1]]        #Most recent time sequence
for i in range(HP // NFS):  #Repeat prediction
    P = np.concatenate([P[:, NFS:], YH], axis = 1)
    YH = cnnr.predict(P)
    PTS.append(YH)
PTS = np.hstack(PTS).transpose((1, 0, 2))
A = np.vstack([A, PTS]) #Combine predictions with original data
A = np.squeeze(A) * SV  #Remove unittime dimension and rescale
C = np.squeeze(C) * SV


CI = list(range(C.shape[0]))
AI = list(range(C.shape[0] + PTS.shape[0] - HP))
NDP = PTS.shape[0] #Number of days predicted
for i, cli in enumerate(cl):
    fig, ax = mpl.subplots(figsize = (16 / 1.5, 10 / 1.5))
    hind = i * len(CN) + CN.index('high')
    ax.plot(CI[-4 * HP:], C[-4 * HP:, hind], label = 'Actual')
    ax.plot(AI[-(NDP + 1):], A[-(NDP + 1):, hind], '--', label = 'Prediction')
    ax.legend(loc = 'upper left')
    ax.set_title(cli + ' (High)')
    ax.set_ylabel('USD')
    ax.set_xlabel('Time')
    ax.axes.xaxis.set_ticklabels([])
    mpl.show()