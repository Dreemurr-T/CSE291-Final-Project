import torch
import torch.nn as nn
from model import RaMBO_backbone
# from fast_soft_sort.pytorch_ops import soft_sort, soft_rank
# from DiffSortFunction import DiffSortFunction


class SortingRegressionModel(nn.Module):
    def __init__(self, regularization_strength=1.0):
        super().__init__()
        self.regularization_strength = regularization_strength
        self.linear = nn.Linear(3, 1)  # Assume input has 3 features

    def forward(self, x):
        sorted_x = DiffSortFunction.apply(x)
        # sorted_x = soft_sort(x, regularization_strength=self.regularization_strength).float()
        output = self.linear(sorted_x)
        return output


def generate_data(num_samples):
    X = torch.randn(num_samples, 3)
    y = X.sum(dim=1, keepdim=True) + torch.randn(num_samples, 1) * 0.1
    return X, y

def generate_test_data(num_samples):
    X = torch.randn(num_samples, 3)
    y = X.sum(dim=1, keepdim=True)
    return X, y

X_train, y_train = generate_data(100)
X_test, y_test = generate_test_data(20)

# Initialize model and loss function
model = SortingRegressionModel(regularization_strength=0.1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:    
        outputs = model(X_test)
        loss = criterion(outputs, y_test)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {loss.item():.4f}')