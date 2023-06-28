import torch
import torch.nn as nn
import torch.nn.functional as F

class MatrixFactorization(torch.nn.Module):
    def __init__(self, config):
        super(MatrixFactorization, self).__init__()
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.dim_mf = config['dim_mf']

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.dim_mf)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.dim_mf)

        self.linear1 = torch.nn.Linear(in_features=self.dim_mf, out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        element_product = torch.mul(user_embedding, item_embedding)
        output = self.linear1(element_product)
        output = self.logistic(output)
        return output

    def init_weight(self):
        pass