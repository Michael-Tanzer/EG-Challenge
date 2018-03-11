from NeuralNetwork import *
from Individual_Deviation import getSTD

def modelVolatility(currency):
    datasetV = getSTD(currency)[8:]
    datasetV, scalerV = normalise(datasetV)
    trainV, testV = splitTrainTest(datasetV, 0.90)
    trainXV, trainYV, testXV, testYV = reshape(trainV, testV)
    modelV = initialiseModel(trainXV, trainYV)
    modelV.save(currency + ".h5")
    predict_length = 20



def saveGraphsFromModelAndDataVol(currency, predict_length):
    dataset = getSTD(currency)[8:]
    dataset, scaler = normalise(dataset)
    trainV, testV = splitTrainTest(dataset, 0.90)
    trainX, trainY, testX, testY = reshape(trainV, testV)

    model = load_model(".\\files\\" + currency + ".h5")


    predictions = predict_sequences_multiple(model, testX[-predict_length:], predict_length)
    plot = plotStuff(dataset, model, scaler, trainX, trainY, testX, testY, predictions, predict_length)

    lastDate = datetime.datetime.strptime(getLastDate(currency), '%d/%m/%Y')
    date_list = [lastDate - datetime.timedelta(days=x) for x in range(0, len(dataset))][::-1]
    date_list.extend([lastDate + datetime.timedelta(days=x) for x in range(1, predict_length + 1)])

    dates = [date.strftime('%d/%m/%y') for date in date_list]

    plot.xticks(range(len(dataset) + predict_length)[::(len(dataset) + predict_length)//6], dates[::(len(dataset) + predict_length)//6])
    plot.xlabel('Day')
    plot.ylabel('Closing Value')
    plot.title(currency[0].upper() + currency[1:].lower() + " Prediction")

    plot.savefig(".\\graphs_vol\\" + currency + ".png")
    plot.close()

if __name__ == "__main__":
    i = 0
    for currency in allCurrencies():
        saveGraphsFromModelAndDataVol(currency, 10)
        print("done " + str(i + 1) + "/" + str(len(allCurrencies())))
        i += 1