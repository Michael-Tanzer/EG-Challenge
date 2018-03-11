from NeuralNetwork import *
from IndividualDeviationForNN import getSTD


def modelVolatility(currency):
    # creates the neural network model for volatility

    datasetV = getSTD(currency)[8:]
    datasetV, scalerV = normalise(datasetV)
    trainV, validationV, testV = splitTrainTest(datasetV, 0.80, 0.10)
    trainXV, trainYV, testXV, testYV, validationXV, validationYV = reshape(trainV, testV,validationV)
    modelV = initialiseModel(trainXV, trainYV, testXV, testYV, validationXV, validationYV)
    modelV.save(".\\files_vol\\" + currency + ".h5")


def saveGraphsFromModelAndDataVol(currency, predict_length):
    # gets data for graph
    dataset = getSTD(currency)[8:]
    dataset, scaler = normalise(dataset)
    trainV, validationV, testV = splitTrainTest(dataset, 0.80, 0.10)
    trainX, trainY, testX, testY, validationXV, validationYV = reshape(trainV, testV, validationV)

    # loads model from file
    model = load_model(".\\files_vol\\" + currency + ".h5")

    # makes predictions
    predictions = predict_sequences_multiple(model, testX[-predict_length:], predict_length)
    plot = plotStuff(dataset, model, scaler, trainX, trainY, testX, testY, predictions, predict_length, validationXV, validationYV)

    # calculates dates range for the graph
    lastDate = datetime.datetime.strptime(getLastDate(currency), '%d/%m/%Y')
    date_list = [lastDate - datetime.timedelta(days=x) for x in range(0, len(dataset))][::-1]
    date_list.extend([lastDate + datetime.timedelta(days=x) for x in range(1, predict_length + 1)])

    dates = [date.strftime('%d/%m/%y') for date in date_list]

    # changes x axis ticks to the dates range
    plot.xticks(range(len(dataset) + predict_length)[::(len(dataset) + predict_length)//6], dates[::(len(dataset) + predict_length)//6])
    plot.xlabel('Day')
    plot.ylabel('Volatility over the previous 7 days')
    plot.title(currency[0].upper() + currency[1:].lower() + " Volatility Prediction")
    #plot.show()
    # saves the plot to a file
    plot.savefig(".\\graphs_vol\\" + currency + ".png")
    plot.close()



if __name__ == "__main__":
    # currencies = allCurrencies()
    #
    # i = 0
    # for currency in currencies:
    #     if i >= 180:
    #         modelVolatility(currency)
    #         print("model " + str(i+1) + "/" + str(len(currencies)))
    #     i+=1
    #
    # print("\n\n  --- DONE WITH MODELS --- \n\n")
    #
    # i = 0
    # for currency in currencies:
    #     if i >= 140:
    #         saveGraphsFromModelAndDataVol(currency, 10)
    #         print("done " + str(i + 1) + "/" + str(len(currencies)))
    #     i += 1

    saveGraphsFromModelAndDataVol('bitcoin', 10)