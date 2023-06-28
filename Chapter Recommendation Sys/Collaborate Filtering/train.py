import pandas as pd
from sklearn import model_selection
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics,preprocessing
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

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



## model
class RecSysModel(nn.Module):
	def __init__(self,num_users,num_movies,emb_size = 100,n_hidden = 10):
		super().__init__()
		self.user_embed = nn.Embedding(num_users,emb_size)
		self.movie_embed = nn.Embedding(num_movies,emb_size)
		self.linear1 = nn.Linear(emb_size * 2,n_hidden)
		self.out = nn.Linear(n_hidden,1)
		self.dropout = nn.Dropout(0.1)

	def forward(self,user,movie):
		user_embeds = self.user_embed(user)
		movie_embeds = self.movie_embed(movie)
		output = F.relu(torch.cat([user_embeds,movie_embeds],dim = 1))
		output = self.dropout(output)
		output = F.relu(self.linear1(output))
		output = self.out(output)

		return output

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

def train(model,epochs=10,lr = 0.02,wd=0,unsqueeze = False):
	df = pd.read_csv('../../../datasets/predict_moview_ratings/train_v2.csv')

	df_train,df_valid = model_selection.train_test_split(df,test_size = 0.1, random_state = 42,stratify = df.rating.values)
	df_train,len_encoding = encode_data(df_train)
	df_valid,_ = encode_data(df_valid,df_train)

	train_ds = moviedataset(df_train)
	valid_ds = moviedataset(df_valid)
	train_dl = DataLoader(train_ds,batch_size = 64,shuffle=True)
	valid_dl = DataLoader(valid_ds,batch_size = 64)

	model = model(len_encoding[0],len_encoding[1]).cuda()
	optimizer = torch.optim.Adam(model.parameters(),lr = lr,weight_decay = wd)
	for i in range(epochs):
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
	train(RecSysModel,unsqueeze = True)
