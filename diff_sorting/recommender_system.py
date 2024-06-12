import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Step 1: 创建示例数据
data = {
    'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
    'item_id': [1, 2, 3, 1, 3, 2, 3, 4, 2, 4],
    'rating': [5, 3, 4, 4, 5, 5, 4, 2, 4, 5]
}
df = pd.DataFrame(data)

# Step 2: 准备数据集
user_ids = df['user_id'].unique()
item_ids = df['item_id'].unique()
user_id_map = {id: i for i, id in enumerate(user_ids)}
item_id_map = {id: i for i, id in enumerate(item_ids)}
df['user_id'] = df['user_id'].map(user_id_map)
df['item_id'] = df['item_id'].map(item_id_map)
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Step 3: 定义排序层
class SortingLayer(nn.Module):
    def __init__(self):
        super(SortingLayer, self).__init__()
    
    def forward(self, x):
        sorted_x, _ = torch.sort(x, dim=-1)
        return sorted_x

# Step 4: 定义推荐模型
class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(NeuralCollaborativeFiltering, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.sorting = SortingLayer()
        self.fc4 = nn.Linear(32, 1)
    
    def forward(self, user_id, item_id):
        user_embed = self.user_embedding(user_id)
        item_embed = self.item_embedding(item_id)
        concatenated = torch.cat([user_embed, item_embed], dim=-1)
        x = F.relu(self.fc1(concatenated))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.sorting(x)
        x = self.fc4(x)
        return x

# Step 5: 训练模型
num_users = len(user_id_map)
num_items = len(item_id_map)
embedding_dim = 32
model = NeuralCollaborativeFiltering(num_users, num_items, embedding_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_users = torch.tensor(train_data['user_id'].values, dtype=torch.long)
train_items = torch.tensor(train_data['item_id'].values, dtype=torch.long)
train_ratings = torch.tensor(train_data['rating'].values, dtype=torch.float32)

num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(train_users, train_items).squeeze()
    loss = criterion(outputs, train_ratings)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

# Step 6: 评估模型
model.eval()
test_users = torch.tensor(test_data['user_id'].values, dtype=torch.long)
test_items = torch.tensor(test_data['item_id'].values, dtype=torch.long)
test_ratings = torch.tensor(test_data['rating'].values, dtype=torch.float32)

with torch.no_grad():
    predictions = model(test_users, test_items).squeeze()
    test_loss = criterion(predictions, test_ratings)
    print(f'Test Loss: {test_loss.item():.4f}')
