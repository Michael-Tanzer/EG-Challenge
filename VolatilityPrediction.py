from NeuralNetwork import *
from Individual_Deviation import getSTD

currency = 'peercoin'
datasetV = getSTD(currency)[8:]
# print(datasetV.type)
datasetV, scalerV = normalise(datasetV)
trainV, testV = splitTrainTest(datasetV, 0.90)
trainXV, trainYV, testXV, testYV = reshape(trainV, testV)

modelV = initialiseModel(trainXV, trainYV)
modelV.save(currency + ".h5")

predict_length = 20
# print(scalerV.inverse_transform(testXV[-10:]))
predictions = predict_sequences_multiple(modelV, testXV[-predict_length:], predict_length)
# print([scalerV.inverse_transform(item) for item in predictions])

plotStuff(datasetV, modelV, scalerV, trainXV, trainYV, testXV, testYV, predictions, predict_length)
print([x for x in scalerV.inverse_transform([predictions])][0])
# plt.plot(range(10), [x for x in scalerV.inverse_transform([predictions])][0])
# plt.show()