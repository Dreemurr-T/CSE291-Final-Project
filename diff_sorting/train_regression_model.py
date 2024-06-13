import torch
import torch.nn as nn
from DiffSortFunction import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class SortingRegressionModel(nn.Module):
    def __init__(self, regularization_strength=1.0, device=device):
        super().__init__()
        self.device = device
        self.regularization_strength = regularization_strength
        self.linear_1 = nn.Linear(3, 1).to(device)
        self.linear_2 = nn.Linear(1, 1).to(device)

    def forward(self, x):
        x = self.linear_1(x)
        x = x.reshape(-1)
        output = DiffSortFunction.apply(x, 0, self.regularization_strength)
        output = output.unsqueeze(1)
        output = self.linear_2(output)
        return output


# -------------------------------- Generate some data -------------------------------- 
def generate_data(num_samples, device=device):
    X = torch.randn(num_samples, 3, device=device)
    y = X.sum(dim=1, keepdim=True) + torch.randn(num_samples, 1, device=device) * 0.1
    return X, y

def generate_test_data(num_samples, device=device):
    X = torch.randn(num_samples, 3, device=device)
    y = X.sum(dim=1, keepdim=True)
    return X, y

X_train, y_train = generate_data(100, device)
X_test, y_test = generate_test_data(20, device)


# -------------------------------- Init model -------------------------------- 
model = SortingRegressionModel(regularization_strength=0.1, device=device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train.to(device))
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            outputs = model(X_test)
            test_loss = criterion(outputs, y_test)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')
