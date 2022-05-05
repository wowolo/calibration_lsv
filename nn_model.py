import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class PARAM():

    def __init__(self) -> None:

        self.N = 20
        self.max_T = 1 # maximal maturity -> time steps: max_T/N

        self.NN_stacks = 3
        self.NN_input = 4 # price S, time t, time to maturity T-t, log moneyness log(S/K), Brownian motion increment dB
        self.NN_varwidth = 30
        self.NN_fixedwidth = 5
        self.NN_output = 2 # leverage, hedge




class NeuralNetwork(nn.Module):



    def __init__(self, PARAM) -> None:

        super(NeuralNetwork, self).__init__()

        self.PARAM = PARAM

        self.neural_network = nn.Sequential() # to do: might want to change initialization - right now uniform initialization of weights and biase...

        for i in range(PARAM.NN_stacks):

            if i == 0: 
                input_width = PARAM.NN_input
                output_width = PARAM.NN_fixedwidth

            elif i == PARAM.NN_stacks - 1:
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


    
    def forward(self, x_dict, detach=None):
        """ x_dict: {(K,T,0): [S_0, K, T, [BMincrement]], ..., (K,T,len(x)): [S_0, K, T, N, [BMincrement]]}
                S_0: The initial price of the asset
                K: The strike of the option
                T: The maturity of the option
                BMincrement: List of N independent realizations of Gaussian(0, 1)
        """
        
        x_keys = list(x_dict.keys())
        x = list(x_dict.values())
        S_0, K, T, BMincrement = x[:, 0], x[:, 1], x[:, 2], x[:, 3] 

        # data check: check that N BMincrements are given
        if len(x[0][-1]) != self.PARAM.N:
            return RuntimeError('The number of given Brownian motion increments ({}) does not match with \
                the number of discretization steps ({}).'.format(len(x[0][-1]), self.PARAM.N))


        leverage, hedge = self._detach_network(detach)


        # recursive computations with N discretizations
        for step in range(self.PARAM.N):
            

        




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



     def loss(self, option_dict, forward_hedge_detached, forward_leverage_detached, x_keys):
        
        loss = self._loss_locvol(option_dict, forward_hedge_detached, x_keys) \
            + self._loss_hedge(option_dict, forward_leverage_detached)

        return loss



    def train(self, option_dict, x_dict):
        # might want to implement with batch size later...
        # use loss with self.optimizer.step() and loss.backward()
        loss_



    def test(self):


# used to generate BM increments for input data
def BMincrement(PARAM):
    BMincr = torch.normal(0, 1) 
    return BMincr
