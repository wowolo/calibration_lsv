# create data dataframes
import numpy as np
import pandas as pd


# used to generate BM increments for input data
def BMincrements(N):
    incr = np.random.normal(0, 1, size=(N,))
    incr = list(incr) 
    return incr



def create_data(option_price, option_data, mc_paths, param):
    """
    option_price - dict: integer keys (matching with option_data keys) and option prices as values
    option_data - dict: integer keys (matching with option_price keys) and values of the format
    [S_0, K, T] (i.e., initial price, strike, maturity)
    mc_path - int: number of monte carlo paths
    param - class: parameter class
    """
    len_data = len(option_price)

    col_labels = ['ID_' + str(i) for i in range(len_data * mc_paths)]
    incr_labels = ['incr_' + str(i) for i in range(param.N)]
    Xdata = pd.DataFrame(index=['S_0', 'K', 'T'] + incr_labels, columns=col_labels)
    ydata = pd.DataFrame(index=['Option Price'], columns=col_labels)    


    for n_mc in range(mc_paths):
        mc_path = BMincrements(param.N)
        start_ID = n_mc * len_data
        for i, id in enumerate(option_price):
            ydata.loc[:, 'ID_' + str(start_ID + i)] = option_price[id]
            Xdata.loc[:, 'ID_' + str(start_ID + i)] = option_data[id] + mc_path

    Xdata = Xdata.T
    Xdata.index.name = 'Samples'
    ydata = ydata.T
    ydata.index.name = 'Samples'

    return Xdata, ydata