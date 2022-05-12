import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import nn_util



class NeuralNetwork(nn.Module):

    def __init__(self, PARAM) -> None:

        super(NeuralNetwork, self).__init__()

        self.PARAM = PARAM

        self.neural_network = nn.Sequential() # to do: might want to change initialization - right now uniform initialization of weights and biase...

        for i in range(PARAM.NN_stacks):

            if i == 0: # first stack
                input_width = PARAM.NN_input
                output_width = PARAM.NN_fixedwidth

            elif i == PARAM.NN_stacks - 1: # last stack
                input_width = PARAM.NN_fixedwidth
                output_width = PARAM.NN_output

            else:
                input_width = PARAM.NN_fixedwidth
                output_width = PARAM.NN_fixedwidth

            variable_width = PARAM.NN_varwidth

            new_stack = self._core_stack(input_width, output_width, variable_width)
            self.neural_network = nn.Sequential(
                self.neural_network, 
                new_stack
                )
        
        self.optimizer = torch.optim.Adam(self.neural_network.parameters()) # torch optim Adam of self.neural_network



    def _core_stack(self, input_width, output_width, variable_width):

        stack = nn.Sequential(
            nn.Linear(input_width, variable_width),
            nn.ReLU(),
            nn.Linear(variable_width, output_width)
            )

        return stack



    def _detach_network(self, detach):

        if detach == 'leverage':
            leverage = lambda x: self.neural_network(x).detach()[:, [0]] # ouput 0 of NN
            hedge = lambda x: self.neural_network(x)[:, [1]]
        elif detach == 'hedge':
            leverage = lambda x: self.neural_network(x)[:, [0]] # ouput 0 of NN
            hedge = lambda x: self.neural_network(x).detach()[:, [1]]
        else:
            leverage = lambda x: self.neural_network(x)[:, [0]] # ouput 0 of NN
            hedge = lambda x: self.neural_network(x)[:, [1]]

        return leverage, hedge


    
    def forward(self, X, detach=None):
        """ 
        X - pd.Dataframe: IDs as index and features [S_0, K, T, BMincrements] as columns
        detach - str: Keywords for detaching within computation graph of neureal network are 
        'leverage' or 'hedge', else no detachment is made.
        """
        
        X = nn_util.to_tensor(X)
        S_0, K, T, BMincrements = nn_util.extract_features(X)

        # data check: check that N BMincrements are given
        if BMincrements.shape[1] != self.PARAM.N:
            return RuntimeError('The number of given Brownian motion increments ({}) does not match with \
                the number of discretization steps ({}).'.format(BMincrements.shape[1], self.PARAM.N))

        # create data for neural networks which will be recursively updated within the network's forward pass
        price = S_0
        hedgepf = torch.zeros_like(S_0)

        # create computation graphs of leverage and hedge
        leverage, hedge = self._detach_network(detach)

        # recursive computations with N discretizations
        for step in range(self.PARAM.N):
            time = torch.ones_like(S_0) * step / self.PARAM.N
            do_comp = (time <= T) # determines if we continue the recursive computations for the respective option with maturitiy T
            step_increment = (BMincrements[:, [step]] * self.PARAM.step_size)
            
            nn_input = torch.cat((price, time, T-time, torch.log(price / K)), dim=1)

            dS = leverage(nn_input) * price * step_increment
            price = price + do_comp * dS
            
            new_hedge = hedge(nn_input) * dS
            hedgepf = hedgepf + do_comp * new_hedge
            
        payoff = ((price - K) + torch.abs(price - K)) / 2
        output = payoff - hedgepf # option_payoff - hedge_portfolio

        return output



    def loss(self, X, y):
        
        forward_hedge_detached = self.forward(X, 'hedge')
        forward_leverage_detached = self.forward(X, 'leverage')

        loss = nn_util.loss_locvol(forward_hedge_detached, y, X) \
            + nn_util.loss_hedge(forward_leverage_detached, y)

        return loss



    def train(self, Xdata, ydata, **kwargs):
        # kwargs: batch_size, epochs, shuffle, num_workers
        if 'epochs' in kwargs.keys():
            epochs = kwargs.pop('epochs')
        else:
            epochs = 1
        
        self.neural_network = self.neural_network.double()

        training_generator = nn_util.DataGenerator(Xdata, ydata, **kwargs)

        # in for loop
        for epoch in range(epochs):
            print('Epoch: ', epoch)
            for X, y in training_generator:
                loss = self.loss(X, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()



    # def test(self):






# # move this part into training
# params = {'batch_size': 64,
#           'shuffle': True,
#           'num_workers': 6}
          
# training_set = LoadData(Xdata, ydata)
# training_generator = torch.utils.data.DataLoader(training_set, **params)