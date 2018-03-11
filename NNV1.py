import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os

from DataProcessForNN import *


# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
look_back = 1
epochs = 10

# fix random seed for reproducibility
numpy.random.seed(7)


def normalise(dataset):
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset.values.reshape(-1,1))
    return dataset, scaler


def splitTrainTest(dataset, train):
    # split into train and test sets
    train_size = int(len(dataset) * train)
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    return train,test


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


def reshape(train, test):
    # reshape into X=t and Y=t+1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    return trainX,trainY,testX,testY


def initialiseModel(look_back, trainX, trainY):
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=epochs, batch_size=1, verbose=2)
    return model


def stuff(model, scaler, trainX, trainY, testX, testY):
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))


    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict
    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(dataset), color='green')
    plt.plot(trainPredictPlot, color='red')
    plt.plot(testPredictPlot, color='orange')
    plt.show()


def predict_sequences_multiple(model, firstValue, length):
    prediction_seqs = []
    curr_frame = firstValue

    for i in range(length):
        predicted = []

        #print(model.predict(curr_frame[numpy.newaxis, :, :]))
        predicted.append(model.predict(curr_frame[numpy.newaxis, :, :])[0, 0])

        curr_frame = curr_frame[0:]
        curr_frame = numpy.insert(curr_frame[0:], i + 1, predicted[-1], axis=0)

        prediction_seqs.append(predicted[-1])

    return prediction_seqs


if __name__ == "__main__":
    currency = 'zcoin'
    dataset = getCurrency(currency)['close']
    dataset,scaler = normalise(dataset)
    train, test = splitTrainTest(dataset, 0.99)
    trainX, trainY, testX, testY = reshape(train, test)

    model = initialiseModel(look_back, trainX, trainY)
    #model.save(currency + ".h5")

    predictedbn = predict_sequences_multiple(model, testX[-1], 10)
    stuff(model,scaler,trainX,trainY,testX,testY)



