import pandas as pd
import torch
import nn_util




def to_tensor(df):
    if isinstance(df, pd.DataFrame):
        X = torch.tensor(df.values, dtype=torch.float64)
    else:
        X = df
    
    return X




def extract_features(X):

    S_0 = X[:, [0]]
    K = X[:, [1]]
    T = X[:, [2]]
    BMincrements = X[:, 3:]

    return S_0, K, T, BMincrements



# create dataset
class Dataset(torch.utils.data.Dataset):

    def __init__(self, Xdata, ydata):
        self.Xdata = to_tensor(Xdata)
        self.ydata = to_tensor(ydata)



    def __len__(self):
        return self.ydata.shape[0]



    def __getitem__(self, index):
        # Select sample
        # Load data and get label
        X = self.Xdata[index]
        y = self.ydata[index]

        return X, y




def DataGenerator(Xdata, ydata, **kwargs):

    dataset = Dataset(Xdata, ydata)
    data_generator = torch.utils.data.DataLoader(dataset, **kwargs)

    return data_generator




def loss_locvol(pred, y, X):

    _, K, T, _ = nn_util.extract_features(X)

    all_K_T = torch.cat((K, T), dim=1)
    unique_K_T = torch.unique(all_K_T, dim=0)

    loss = 0

    for K_T in unique_K_T:
        ind = [torch.equal(temp, K_T) for temp in all_K_T]
        loss = loss + (torch.sum(pred[ind] - y[ind]) / len(ind)) ** 2

    loss = loss / len(unique_K_T) # average over number of samples

    return loss




def loss_hedge(pred, y):
    
    loss = torch.nn.MSELoss()(pred, y)

    return loss
