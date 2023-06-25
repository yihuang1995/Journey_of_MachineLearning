import pandas as pd
from sklearn import model_selection
import torch
import torch.nn as nn
from sklearn import metrics,preprocessing
import numpy

class RecSysModel(nn.Module):
	def __init__(self,num_users,num_movies,emb_size = 100,n_hidden = 10):
		super().__init__()
		self.user_embed = nn.Embedding(num_users,emb_size)
		self.movie_embded = nn.Embedding(num_movies,emb_size)
		self.linear1 = nn.Linear(emb_size * 2,n_hidden)
		self.out = nn.Linear(n_hidden,1)
		self.dropout = nn.Dropout(0.1)

	def forward(self,user,movie):
		user_embeds = self.user_embed(user)
		movie_embeds = self.movie_embed(movie)
		output = F.relu(touch.cat([user_embed,movie_embeds],dim = 1))
		output = self.dropout(output)
		output = F.relu(self.linear1(output))
		output = self.out(output)

		return output

def test_loss(model,df_valid,unsqueeze = False):
	model.eval()
	users = torch.LongTensor(df_valid.user.values)
	movies = torch.LongTensor(df_valid.movie.values)
	ratings = torch.FloatTensor(df_valid.rating.values)
	if unsqueeze:
		ratings = ratings.unsqueeze(1)
	y_hat = model(users,movies)
	loss = F.mse_loss(y_hat,ratings)
	print("test loss %.3f " %loss.item())

def train(model,epochs=10,lr = 0.01,wd=0,unsqueeze = False):
	df = pd.read_csv('../../../datasets/predict_moview_ratings/train_v2.csv')

	lbl_user = preprocessing.LabelEncoder()
	lbl_movie = preprocessing.LabelEncoder()
	df_user = lbl_user.fit_transform(df.user.values)
	df_movie = lbl_movie.fit_transform(df.movie.values)

	df_train,df_valid = model_selection.train_test_split(df,test_size = 0.1, random_state = 42,stratify = df.rating.values)

	model = RecSysModel(len(lbl_user.classes_),len(lbl_movie.classes_))
	optimizer = torch.optim.Adam(model.parameters(),lr = lr,weight_decay = wd)
	model.train()
	for i in range(epochs):
		users = torch.LongTensor(df_train.user.values)
		movies = torch.LongTensor(df_train.movie.values)
		ratings = torch.FloatTensor(df_train.rating.values)
		if unsqueeze:
			ratings = ratings.unsqueeze(1)
		y_hat = model(users,ratings)
		loss = F.mse_loss(y_hat,ratings)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		print(loss.item)
	test_loss(model,df_valid,unsqueeze)


if __name__ == "__main__":
	train(RecSysModel)
