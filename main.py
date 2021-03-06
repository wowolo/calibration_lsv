# main file for executing program
import torch
from create_data import create_data
from nn_model import NeuralNetwork

class PARAM():

    def __init__(self) -> None:

        self.N = 20
        self.maturities = [0.5, 1] 
        self.max_T = max(self.maturities) # time steps: max_T / N & guarantee that maturities are multiple of time steps
        self.step_size = self.max_T / self.N

        # model architecture
        self.NN_stacks = 14
        self.NN_input = 4 # price S, time t, time to maturity T-t, log moneyness log(S/K)
        self.NN_varwidth = 45
        self.NN_fixedwidth = 15
        self.NN_output = 2 # leverage, hedge

        # training
        self.mc_paths = 500
        self.batch_size = 32
        self.epochs = 5

        # data features?

param = PARAM()

# Data
option_price = {0: 0.20042534,
            1: 0.23559685,
            2: 0.16312157,
            3: 0.20771958,
            4: 0.13154241,
            5: 0.18236567}
# S, K, T
option_data = {0: [1, 0.9, 0.5],
            1: [1, 0.9, 1.0],
            2: [1, 1.0, 0.5],
            3: [1, 1.0, 1.0],
            4: [1, 1.1, 0.5],
            5: [1, 1.1, 1.0]}


Xdata, ydata = create_data(option_price, option_data, param)

model = NeuralNetwork(param)
model.train(Xdata, ydata, epochs=param.epochs, batch_size=param.batch_size)