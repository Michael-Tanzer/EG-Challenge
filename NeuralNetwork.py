import datetime
import math
import os

import matplotlib.pyplot as plt
import numpy
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
from keras.models import Sequential, load_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from DataProcessForNN import *

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
look_back = 1
epochs = 100

# fix random seed for reproducibility
#numpy.random.seed(7)


def normalise(dataset):
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset.values.reshape(-1,1))
    return dataset, scaler


def splitTrainTest(dataset, train, validation):
    # split into train and test sets
    train_size = int(len(dataset) * train)
    train, validation, test = dataset[0:train_size, :], dataset[train_size:train_size + int(len(dataset)*validation), :], dataset[train_size + int(len(dataset)*validation):len(dataset):]
    return train, validation, test



def create_dataset(dataset):
    # convert an array of values into a dataset matrix
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


def reshape(train, test, validation):
    # reshape into X=t and Y=t+1
    trainX, trainY = create_dataset(train)
    testX, testY = create_dataset(test)

    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    validationX, validationY = create_dataset(validation)
    validationX = numpy.reshape(validationX, (validationX.shape[0], 1, validationX.shape[1]))

    return trainX,trainY,testX,testY, validationX, validationY


def initialiseModel(trainX, trainY, testX, testY, validationX, validationY):
    # create and fit the LSTM network
    model = Sequential()

    model.add(LSTM(return_sequences=True, input_shape=(None, 1), units=50))
    model.add(Dropout(0.2))

    model.add(LSTM(
        100,
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(units=1))
    model.add(Activation('linear'))

    model.compile(loss='mse', optimizer='rmsprop')

    model.fit(trainX, trainY, batch_size=128, epochs=epochs, verbose=False, validation_data=(validationX, validationY))
    return model


def plotStuff(dataset, model, scaler, trainX, trainY, testX, testY, predict, length, validationX, validationY):
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    validationPredict = model.predict(validationX)

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    validationPredict = scaler.inverse_transform(validationPredict)
    validationY = scaler.inverse_transform([validationY])

    #calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))
    valScore = math.sqrt(mean_squared_error(validationY[0], validationPredict[:,0]))
    print('Validation Score: %.2f RMSE' % (valScore))



    #  shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

    valPredictPlot = numpy.empty_like(dataset)
    valPredictPlot[:, :] = numpy.nan
    try:
        valPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(trainPredict) + len(validationPredict) + 3, :] = validationPredict
    except:
        valPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(trainPredict) + len(validationPredict) + 2,:] = validationPredict
    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    try:
        testPredictPlot[len(trainPredict) + len(validationPredict) + 5:len(dataset) - 1, :] = testPredict
    except:
        testPredictPlot[len(trainPredict) + len(validationPredict) + 4:len(dataset) - 1, :] = testPredict


    # plot baseline and predictions
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.plot(valPredictPlot)
    plt.plot(range(len(dataset), len(dataset) + length), [x for x in scaler.inverse_transform([predict])][0][::-1])
    #plt.show()
    return plt


def predict_sequences_multiple(model, firstValue, length):
    # predicts the next values starting from n given values and a model
    prediction_seqs = []
    curr_frame = firstValue

    for i in range(length):
        predicted = []

        predicted.append(model.predict(curr_frame)[0, 0])

        curr_frame = curr_frame[-length:]
        curr_frame = numpy.insert(curr_frame[-length:], i + 1, predicted[-1], axis=0)

        prediction_seqs.append(predicted[-1])

    return prediction_seqs


def plot_results_multiple(predicted_data, true_data, scaler):
    # small function to plot predicted data vs true_data
    plt.plot(scaler.inverse_transform(true_data.reshape(-1, 1)))
    plt.plot(scaler.inverse_transform(numpy.array(predicted_data).reshape(-1, 1)))
    plt.show()


def getUsefulData(currency):
    # gets all the data needed to for other functions like plotstuff

    datasetV = getCurrency(currency)['close']

    datasetV, scalerV = normalise(datasetV)
    trainV, validationV, testV = splitTrainTest(datasetV, 0.80, 0.10)
    trainXV, trainYV, testXV, testYV, validationXV, validationYV = reshape(trainV, testV, validationV)

    return scalerV, datasetV, trainXV, trainYV, testXV, testYV, validationXV, validationYV


def test():
    # function used for testing during the development

    currency = 'bitcoin'
    datasetV = getCurrency(currency)['close']

    datasetV, scalerV = normalise(datasetV)
    trainV, validationV, testV = splitTrainTest(datasetV, 0.80, 0.10)
    trainXV, trainYV, testXV, testYV, validationXV, validationYV = reshape(trainV, testV, validationV)

    modelV = initialiseModel(trainXV, trainYV, testXV, testYV, validationXV, validationYV)
    #modelV.save(currency + ".h5")

    predict_length = 10

    predictions = predict_sequences_multiple(modelV, testXV[-predict_length:], predict_length)


    plot = plotStuff(datasetV, modelV, scalerV, trainXV, trainYV, testXV, testYV, predictions, predict_length, validationXV, validationYV)
    plot.show()
    #print([x for x in scalerV.inverse_transform([predictions])][0])



def saveGraphsFromModelAndData(currency, predict_length):
    # load the model corresponding to a currency and makes a graph and saves it to a file

    # loads the model
    model = load_model(".\\files\\" + currency + ".h5")

    # gets data for the graph
    scaler, datasetV, trainX, trainY, testX, testY, validationX, validationY = getUsefulData(currency)
    predictions = predict_sequences_multiple(model, testX[-predict_length:], predict_length)
    plot = plotStuff(datasetV, model, scaler, trainX, trainY, testX, testY, predictions, predict_length, validationX, validationY)

    # gets the date range of the data
    lastDate = datetime.datetime.strptime(getLastDate(currency), '%d/%m/%Y')
    date_list = [lastDate - datetime.timedelta(days=x) for x in range(0, len(datasetV))][::-1]
    date_list.extend([lastDate + datetime.timedelta(days=x) for x in range(1, predict_length + 1)])

    dates = [date.strftime('%d/%m/%y') for date in date_list]

    # sets the graph x axis to the dates
    plot.xticks(range(len(datasetV) + predict_length)[::(len(datasetV) + predict_length)//6], dates[::(len(datasetV) + predict_length)//6])
    plot.xlabel('Day')
    plot.ylabel('Closing Value')
    plot.title(currency[0].upper() + currency[1:].lower() + " Prediction")

    # saves the file
    plot.savefig(".\\graphs\\" + currency + ".png")
    plot.close()


if __name__ == "__main__":
    saveGraphsFromModelAndData('bitcoin', 10)
    # flag = False
    # print(allCurrencies())
    # for currency in allCurrencies():
    #     if currency == "tierion":
    #         flag = True
    #         continue
    #
    #     if flag:
    #         datasetV = getCurrency(currency)['close']
    #         # print(datasetV.type)
    #         datasetV, scalerV = normalise(datasetV)
    #         trainV, testV = splitTrainTest(datasetV, 0.95)
    #         trainXV, trainYV, testXV, testYV = reshape(trainV, testV)
    #
    #         modelV = initialiseModel(trainXV, trainYV)
    #         modelV.save(".\\files\\" + currency + ".h5")
    #
    #         print("saved " + currency)
    # i = 0
    # for currency in allCurrencies():
    #     if i >= 134:
    #         saveGraphsFromModelAndData(currency, 10)
    #         print("done " + str(i + 1) + "/" + str(len(allCurrencies())))
    #     i += 1

