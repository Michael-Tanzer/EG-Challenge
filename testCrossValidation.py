from NeuralNetwork import *
import numpy as np


def runNN(X_train_set, Y_train_set, X_test_set, Y_test_set):
    return initialiseModel(X_train_set, Y_train_set, X_test_set, Y_test_set)

my_grid=[2,3,5,10,20,50,100,200,500,1000,5000]
nn_outputs = []
nn_outputs_early = []

for i in my_grid:
    scaler, dataset, X_train, Y_train, X_test, Y_test = getUsefulData('litecoin')
    nn_output_unscaled = runNN(X_train, Y_train, X_test, Y_test)
    nn_output_scaled = runNN(scaler.fit_transform(X_train), Y_train,scaler.fit_transform(X_test), Y_test)
    nn_output_unscaled_early = runNN(X_train, Y_train, X_test, Y_test)
    nn_output_scaled_early = runNN(scaler.fit_transform(X_train), Y_train, scaler.fit_transform(X_test), Y_test)

    nn_outputs.append([i, nn_output_unscaled['train_loss'],nn_output_unscaled['test_loss'],
                       nn_output_scaled['train_loss'], nn_output_scaled['test_loss']])
    nn_outputs_early.append([i, nn_output_unscaled_early['train_loss'],nn_output_unscaled_early['validation_loss'],
                             nn_output_unscaled_early['test_loss'], nn_output_scaled_early['train_loss'],
                             nn_output_scaled_early['validation_loss'], nn_output_scaled_early['test_loss']])

nn_outputs = np.array(nn_outputs)
nn_outputs_early = np.array(nn_outputs_early)