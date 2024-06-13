import torch
import torch.nn as nn
import torch.nn.functional as F
from DiffSortFunction import DiffSortFunction

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# define sorting layer
class SortingLayer(nn.Module):
    def __init__(self, reverse=1, regularization_strength=1.0):
        super(SortingLayer, self).__init__()
        self.reverse = reverse
        self.regularization_strength = regularization_strength
    
    def forward(self, x):
        sorted_x = DiffSortFunction.apply(x, self.reverse, self.regularization_strength)
        return sorted_x


# sefine neural collaborative filtering model
class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, reverse=1, regularization_strength=1.0, device=device):
        super(NeuralCollaborativeFiltering, self).__init__()
        self.device = device
        self.user_embedding = nn.Embedding(num_users, embedding_dim).to(device)
        self.item_embedding = nn.Embedding(num_items, embedding_dim).to(device)
        self.fc1 = nn.Linear(embedding_dim * 2, 128).to(device)
        self.fc2 = nn.Linear(128, 64).to(device)
        self.fc3 = nn.Linear(64, 32).to(device)
        self.fc4 = nn.Linear(32, 1).to(device)
        self.sorting = SortingLayer(reverse=reverse, regularization_strength=regularization_strength)
        self.fc5 = nn.Linear(1, 1).to(device)
    
    def forward(self, user_id, item_id):
        user_embed = torch.index_select(self.user_embedding.weight, 0, user_id)
        item_embed = torch.index_select(self.item_embedding.weight, 0, item_id)
    
        concatenated = torch.cat([user_embed, item_embed], dim=-1)
        x = F.relu(self.fc1(concatenated))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = x.reshape(-1)
        x = self.sorting(x)
        x = x.unsqueeze(1)
        x = self.fc5(x)
        return x