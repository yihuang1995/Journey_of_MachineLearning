import pandas as pd
from sklearn import model_selection
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics,preprocessing
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

## model
class NeuralMF(nn.Module):
	"""concat outputs of mf and mlp then feed into a linear layer"""
	def __init__(self,config):
		super(NeuralMF,self).__init__()
		self.num_users = config['num_users']
		self.num_items = config['num_items']
		self.dim_mf = config['dim_mf']
		self.dim_mlp = config['dim_mlp']

		self.user_embed_mf = nn.Embedding(self.num_users,self.dim_mf)
		self.item_embed_mf = nn.Embedding(self.num_items,self.dim_mf)
		self.user_embed_mlp = nn.Embedding(self.num_users,self.dim_mlp)
		self.item_embed_mlp = nn.Embedding(self.num_items,self.dim_mlp)
		self.fc_layers = torch.nn.ModuleList()
		for idx, (in_size, out_size) in enumerate(zip(config['layers'][:-1], config['layers'][1:])):
			self.fc_layers.append(torch.nn.Linear(in_size, out_size))
			self.fc_layers.append(torch.nn.Dropout(config['drop_out']))
		self.linear1 = torch.nn.Linear(in_features=config['layers'][-1] + config['dim_mf'], out_features=1)
        # self.logistic = torch.nn.Sigmoid()

	def forward(self,user,item):
		user_embed_mlp = self.user_embed_mlp(user)
		item_embed_mlp = self.item_embed_mlp(item)
		user_embed_mf = self.user_embed_mf(user)
		item_embed_mf = self.item_embed_mf(item)

		mlp_vector = torch.cat([user_embed_mlp, item_embed_mlp], dim=-1)  # the concat latent vector
		mf_vector =torch.mul(user_embed_mf, item_embed_mf)

		for idx, _ in enumerate(range(len(self.fc_layers))):
			mlp_vector = self.fc_layers[idx](mlp_vector)
			mlp_vector = torch.nn.ReLU()(mlp_vector)

		output = torch.cat([mlp_vector, mf_vector], dim=-1)
		output = self.linear1(output)
		return output
        # output = self.logistic(output)

	def load_weights(self):
		"""Optional, loading pretained weights for mf and mlp as transfer learning inputs"""
		...