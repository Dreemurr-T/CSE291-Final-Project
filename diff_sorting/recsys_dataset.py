from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from DiffSortFunction import DiffSortFunction


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# load data
def load_data(filename):
    df = pd.read_csv(filename)
    
    user_ids = df['user_id'].unique()
    item_ids = df['item_id'].unique()
    train_data, test_data = train_test_split(df, test_size=0.15, random_state=42)
    
    return train_data, test_data, user_ids, item_ids

train_data, test_data, user_ids, item_ids = load_data('sample_data.csv')


class RatingDataset(Dataset):
    """Dataset class for loading data."""
    def __init__(self, user_ids, item_ids, ratings):
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.ratings = ratings

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.ratings[idx]


train_user_ids = torch.tensor(train_data['user_id'].values, dtype=torch.long).to(device)
train_item_ids = torch.tensor(train_data['item_id'].values, dtype=torch.long).to(device)
train_ratings = torch.tensor(train_data['rating'].values, dtype=torch.float32).to(device)

# Create dataset
train_dataset = RatingDataset(train_user_ids, train_item_ids, train_ratings)

test_user_ids = torch.tensor(test_data['user_id'].values, dtype=torch.long).to(device)
test_item_ids = torch.tensor(test_data['item_id'].values, dtype=torch.long).to(device)
test_ratings = torch.tensor(test_data['rating'].values, dtype=torch.float32).to(device)

test_dataset = RatingDataset(test_user_ids, test_item_ids, test_ratings)