import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

class PARAM():

    def __init__(self) -> None:

        self.N = 20
        self.maturities = [0.5, 1] 
        self.max_T = max(self.maturities) # time steps: max_T / N & guarantee that maturities are multiple of time steps
        self.step_size = self.max_T / self.N

        self.NN_stacks = 3
        self.NN_input = 4 # price S, time t, time to maturity T-t, log moneyness log(S/K)
        self.NN_varwidth = 30
        self.NN_fixedwidth = 5
        self.NN_output = 2 # leverage, hedge




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
        
        self.optimizer = torch.optim.Adam(self.neural_network.parameters) # torch optim Adam of self.neural_network



    def _core_stack(self, input_width, output_width, variable_width):

        stack = nn.Sequential(
            nn.Linear(input_width, variable_width),
            nn.ReLU(),
            nn.Linear(variable_width, output_width)
            )

        return stack



    def _detach_network(self, detach):

        if detach == 'leverage':
            leverage = lambda x: self.neural_network(x).detach()[0] # ouput 0 of NN
            hedge = lambda x: self.neural_network(x)[1]
        elif detach == 'hedge':
            leverage = lambda x: self.neural_network(x)[0] # ouput 0 of NN
            hedge = lambda x: self.neural_network(x).detach()[1]
        else:
            leverage = lambda x: self.neural_network(x)[0] # ouput 0 of NN
            hedge = lambda x: self.neural_network(x)[1]
        

        return leverage, hedge


# TODO: change from x_dict to new data format!
    
    def forward(self, x_dict, detach=None):
        """ x_dict: {ID_0: [S_0, K, T, [BMincrement]], ..., ID_len(Xdata)): [S_0, K, T, [BMincrement]]}
                S_0: The initial price of the asset
                K: The strike of the option
                T: The maturity of the option
                BMincrement: List of N independent realizations of Gaussian(0, 1)
        """
        
        x = list(x_dict.values())
        S_0, K, T, BMincrement = x[:, 0], x[:, 1], x[:, 2], x[:, 3] 


        # data check: check that N BMincrements are given
        if len(x[0][-1]) != self.PARAM.N:
            return RuntimeError('The number of given Brownian motion increments ({}) does not match with \
                the number of discretization steps ({}).'.format(len(x[0][-1]), self.PARAM.N))


        # create data for neural networks which will be recursively updated within the network's forward pass
        price = S_0
        hedgepf = 0
        time = 0

        # create computation graphs of leverage and hedge
        leverage, hedge = self._detach_network(detach)


        # recursive computations with N discretizations
        for step in range(self.PARAM.N):
            time = (step + 1) / self.PARAM.N
            do_comp = int(time <= T) # determines if we continue the recursive computations for the respective option with maturitiy T
            step_increment = (BMincrement[:, step] * self.PARAM.step_size)
            
            dS = leverage(price, time, T-time, torch.log(price / K)) * price * step_increment
            price += do_comp * dS
            
            new_hedge = hedge(price, time, T-time, torch.log(price / K)) * step_increment
            hedgepf += do_comp * new_hedge
            
        payoff = ((price - K) + torch.abs(price - K)) / 2
        output = payoff - hedgepf # option_payoff - hedge_portfolio

        return output



    def _loss_locvol(self, option_dict, forward_hedge_detached, x_keys):
        # need to address all computations with equivalent option - C(K, T)
        # ideas: dictionary with keys (K,T)?
        all_keys = [l[:2] for l in x_keys]
        unique_keys = list(np.unique(all_keys))

        loss = 0

        for key in unique_keys:
            ind = [key == l for l in all_keys]
            loss += (torch.sum(option_dict.values()[ind] - forward_hedge_detached[ind]) ** 2) / len(ind)
        
        loss /= len(unique_keys) # average over number of samples

        return loss



    def _loss_hedge(self, option_dict, forward_leverage_detached):
        
        loss = nn.MSELoss()(option_dict.values(), forward_leverage_detached)

        return loss



    def loss(self, option_dict, x_dict):
        
        forward_hedge_detached = self.forward(x_dict, 'hedge')
        forward_leverage_detached = self.forward(x_dict, 'leverage')

        x_keys = x_dict.keys()

        loss = self._loss_locvol(option_dict, forward_hedge_detached, x_keys) \
            + self._loss_hedge(option_dict, forward_leverage_detached)

        return loss



    def train(self, option_dict, x_dict):
        # might want to implement with batch size later...
        # use loss with self.optimizer.step() and loss.backward()
        # batch_size = 64

        # in for loop
        loss = self.loss(option_dict, x_dict)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



    # def test(self):


# used to generate BM increments for input data
def BMincrement(PARAM):
    return torch.normal(0, 1, size=(PARAM.N,)) 


# create adapted "data structures"
class LoadData(torch.utils.data.Dataset):

    def __init__(self, Xdata, ydata):
        self.Xdata = Xdata # pd.DataFrame with IDs as index and features as column
        self.ydata = ydata # pd.DataFrame with IDs as index and option price as (only) column

    def __len__(self):
        return self.ydata.shape[0]

    def __getitem__(self, index):
        # Select sample
        if isinstance(index, int): # guarantee list format of IDs
            print('needed index to list in LoadData')
            index = [index]

        # Load data and get label
        X = self.Xdata.iloc[index]
        y = self.ydata.iloc[index]

        return X, y

# Data
option_price = {'0': 0.20042534,
            '1': 0.23559685,
            '2': 0.16312157,
            '3': 0.20771958,
            '4': 0.13154241,
            '5': 0.18236567}
# S, K, T
option_data = {'0': [1, 0.9, 0.5],
            '1': [1, 0.9, 1.0],
            '2': [1, 1.0, 0.5],
            '3': [1, 1.0, 1.0],
            '4': [1, 1.1, 0.5],
            '5': [1, 1.1, 1.0]}


mc_paths = 2
len_data = len(option_price)

ydata = pd.DataFrame(index=['Option Price'])
Xdata = pd.DataFrame(index=['S_0', 'K', 'T', 'BMincrements'])


for n_mc in range(mc_paths):
    mc_path = BMincrement(PARAM())
    start_ID = n_mc * len_data
    for i, id in enumerate(option_price):
        ydata['ID_' + str(start_ID + i)] = option_price[id]
        Xdata['ID_' + str(start_ID + i)] = option_data[id] + [mc_path]

Xdata = Xdata.T
Xdata.index.name = 'Samples'
ydata = ydata.T
ydata.index.name = 'Samples'
        



params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6}
          
training_set = LoadData(Xdata, ydata)
training_generator = torch.utils.data.DataLoader(training_set, **params)

# validation_set = LoadData(partition['validation'], labels)
# validation_generator = torch.utils.data.DataLoader(validation_set, **params)
