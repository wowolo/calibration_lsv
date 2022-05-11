import torch




def df_to_tensor(df):

    X = torch.tensor(df.values, dtype=torch.float64)
    
    return X




def extract_features(X):

    S_0 = X[:, 0]
    K = X[:, 1]
    T = X[:, 2]
    BMincrements = X[:, 3:]

    return S_0, K, T, BMincrements



# create dataset
class Dataset(torch.utils.data.Dataset):

    def __init__(self, Xdata, ydata):
        self.Xdata = df_to_tensor(Xdata.values)
        self.ydata = df_to_tensor(ydata)



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
   
    helper = X.loc[:,['K', 'T']].reset_index().set_index(['K', 'T']) # has IDs as values
    unique_keys = helper.index.unique()

    loss = 0

    for key in unique_keys:
        ind = list(helper.index.isin([key]))
        loss += (torch.sum(y.values()[ind] - pred[ind])/ len(ind)) ** 2
    
    loss /= len(unique_keys) # average over number of samples

    return loss



def loss_hedge(pred, y):
    
    loss = torch.nn.MSELoss()(y.values(), pred)

    return loss
