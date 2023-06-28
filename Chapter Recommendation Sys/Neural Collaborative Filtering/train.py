import pandas as pd
from sklearn import model_selection
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics,preprocessing
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from nmf import NeuralMF

class moviedataset(Dataset):
    def __init__(self,df):
        self.users = df.user.values
        self.movies = df.movie.values
        self.ratings = df.rating.values
    def __len__(self):
        return len(self.users)
    def __getitem__(self,index):
        return self.users[index],self.movies[index],self.ratings[index]

## encoding
def proc_col(col, train_col=None):
    """Encodes a pandas column with continous ids. 
    """
    if train_col is not None:
        uniq = train_col.unique()
    else:
        uniq = col.unique()
    name2idx = {o:i for i,o in enumerate(uniq)}
    return name2idx, np.array([name2idx.get(x, -1) for x in col]), len(uniq)


def encode_data(df, train=None):
    """ Encodes rating data with continous user and movie ids. 
    If train is provided, encodes df with the same encoding as train.
    """
    df = df.copy()
    len_encoding = []
    for col_name in ["user", "movie"]:
        train_col = None
        if train is not None:
            train_col = train[col_name]
        _,col,len_uniq = proc_col(df[col_name], train_col)
        df[col_name] = col
        df = df[df[col_name] >= 0]
        len_encoding.append(len_uniq)
    return df,len_encoding


def valid_loss(model,valid_dl,unsqueeze = False):
    model.eval()
    total_loss = 0
    total = 0
    for valid_user,valid_movie,valid_rating in valid_dl:    
        users = torch.LongTensor(valid_user).cuda()
        movies = torch.LongTensor(valid_movie).cuda()
        ratings = torch.FloatTensor(valid_rating.float()).cuda()
        if unsqueeze:
            ratings = ratings.unsqueeze(1)
        y_hat = model(users,movies)
        loss = F.mse_loss(y_hat,ratings)
        total_loss +=loss.item()*ratings.size(0)
        batch = users.size(0)
        total += batch
    print(f"Valid loss:{total_loss/total:.3f} \t ")

def train(model,config,unsqueeze = False):
    df = pd.read_csv('../../../../datasets/predict_moview_ratings/train_v2.csv')

    df_train,df_valid = model_selection.train_test_split(df,test_size = 0.1, random_state = 42,stratify = df.rating.values)
    df_train,len_encoding = encode_data(df_train)
    df_valid,_ = encode_data(df_valid,df_train)

    train_ds = moviedataset(df_train)
    valid_ds = moviedataset(df_valid)
    train_dl = DataLoader(train_ds,batch_size = config['batch_size'],shuffle=True)
    valid_dl = DataLoader(valid_ds,batch_size = config['batch_size'])

    #model = model(len_encoding[0],len_encoding[1]).cuda()
    model = model(config).cuda()
    optimizer = torch.optim.Adam(model.parameters(),lr = config['lr'],weight_decay = config['wd'])
    for i in range(config['num_epoch']):
        model.train()
        total_loss = 0
        total = 0
        for train_user,train_movie,train_rating in tqdm(train_dl):
            users = torch.LongTensor(train_user).cuda()
            movies = torch.LongTensor(train_movie).cuda()
            ratings = torch.FloatTensor(train_rating.float()).cuda()
            if unsqueeze:
                ratings = ratings.unsqueeze(1)
            y_hat = model(users,movies)
            loss = F.mse_loss(y_hat,ratings)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()*ratings.size(0)
            total +=ratings.size(0)
        print(f"Train loss:{total_loss/total:.3f} \t ")
        valid_loss(model,valid_dl,unsqueeze = True)



if __name__ == "__main__":
    neumf_config = {'alias': 'pretrain_neumf_factor8neg4',
                'num_epoch': 200,
                'batch_size': 1024,
                'lr': 1e-3,
                'wd': 0,
                'num_users': 6040,
                'num_items': 3660,
                'dim_mf': 16,
                'dim_mlp': 16,
                'num_negative': 4,
                'layers': [32,16,8]}  # layers[0] is the concat of latent user vector & latent item vector
    train(NeuralMF,neumf_config,unsqueeze = True)