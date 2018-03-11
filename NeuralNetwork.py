import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os

from DataProcess import *


# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
look_back = 1
epochs = 100

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
    return train, test


# convert an array of values into a dataset matrix
def create_dataset(dataset):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


def reshape(train, test):
    # reshape into X=t and Y=t+1
    trainX, trainY = create_dataset(train)
    testX, testY = create_dataset(test)

    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    return trainX,trainY,testX,testY


def initialiseModel(trainX, trainY):
    # create and fit the LSTM network
    model = Sequential()

    model.add(LSTM(return_sequences=True, input_shape=(None, 1), units=50))
    model.add(Dropout(0.2))

    model.add(LSTM(
        100,
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        units=1))
    model.add(Activation('linear'))

    model.compile(loss='mse', optimizer='rmsprop')

    model.fit(trainX, trainY, batch_size=128, epochs=epochs, validation_split=0.05)
    return model


def stuff(dataset, model, scaler, trainX, trainY, testX, testY, predict):
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    #calculate root mean squared error
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
    #plt.plot(scaler.inverse_transform(dataset))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)

    plt.plot(range(len(dataset), len(dataset) + 10), [x for x in scalerV.inverse_transform([predictions])][0][::-1])
    plt.show()


def predict_sequences_multiple(model, firstValue, length):
    prediction_seqs = []
    curr_frame = firstValue

    for i in range(length):
        predicted = []

        #print(model.predict(curr_frame[numpy.newaxis, :, :]))
        #predicted.append(model.predict(curr_frame[numpy.newaxis, :, :])[0, 0])
        predicted.append(model.predict(curr_frame)[0, 0])

        curr_frame = curr_frame[-10:]
        curr_frame = numpy.insert(curr_frame[-10:], i + 1, predicted[-1], axis=0)

        prediction_seqs.append(predicted[-1])

    return prediction_seqs


def plot_results_multiple(predicted_data, true_data,length, scaler):
    plt.plot(scaler.inverse_transform(true_data.reshape(-1, 1)))
    plt.plot(scaler.inverse_transform(numpy.array(predicted_data).reshape(-1, 1)))
    plt.show()


if __name__ == "__main__":
    currency = 'aion'
    datasetV = getCurrency(currency)['close']
    #print(datasetV.type)
    datasetV,scalerV = normalise(datasetV)
    trainV, testV = splitTrainTest(datasetV, 0.90)
    trainXV, trainYV, testXV, testYV = reshape(trainV, testV)

    modelV = initialiseModel(trainXV, trainYV)
    #model.save(currency + ".h5")



    predict_length = 10
    #print(scalerV.inverse_transform(testXV[-10:]))
    predictions = predict_sequences_multiple(modelV, testXV[-10:], predict_length)
    #print([scalerV.inverse_transform(item) for item in predictions])

    stuff(datasetV, modelV, scalerV, trainXV, trainYV, testXV, testYV, predictions)
    print([x for x in scalerV.inverse_transform([predictions])][0])
    #plt.plot(range(10), [x for x in scalerV.inverse_transform([predictions])][0])
    #plt.show()

