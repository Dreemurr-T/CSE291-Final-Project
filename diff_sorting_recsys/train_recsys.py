import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from recsys_model import NeuralCollaborativeFiltering
from torch.utils.data import Dataset, DataLoader
from recsys_dataset import *
import warnings
warnings.filterwarnings('ignore')

# Init
num_users = user_ids.max() + 1
num_items = item_ids.max() + 1
embedding_dim = 32

model = NeuralCollaborativeFiltering(num_users, num_items, embedding_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# DataLoader
batch_size = 64
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (users, items, ratings) in enumerate(train_data_loader):
        users, items, ratings = users.to(device), items.to(device), ratings.to(device)
        
        optimizer.zero_grad()
       
        outputs = model(users, items).squeeze()
        loss = criterion(outputs, ratings)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Train Loss: {total_loss/len(train_data_loader):.4f}', end=' ')

    model.eval()
    with torch.no_grad():
        test_loss = 0
        for batch_idx, (test_user_ids, test_item_ids, test_ratings) in enumerate(test_data_loader):
            test_user_ids, test_item_ids, test_ratings = test_user_ids.to(device), test_item_ids.to(device), test_ratings.to(device)
            test_outputs = model(test_user_ids, test_item_ids).squeeze()
            test_loss += criterion(test_outputs, test_ratings).item()
        print(f'Test Loss: {test_loss/len(test_data_loader):.4f}')
