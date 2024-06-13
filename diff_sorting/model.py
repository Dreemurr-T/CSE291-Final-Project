import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights
import torchvision.models as models
import torch.nn.functional as F

def _initialize_weights(*layers):
    for m in layers:
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class RaMBO_backbone(nn.Module):
    def __init__(self, embedding_dim=512):
        super(RaMBO_backbone, self).__init__()
        # Load pre-trained ResNet50 model
        self.resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # Replace the final fully connected layer with an embedding layer
        num_features = self.resnet50.fc.in_features
        added_embedding = nn.Linear(num_features, embedding_dim)

        _initialize_weights([added_embedding])

        self.resnet50.fc = added_embedding

    def forward(self, x):
        x = self.resnet50(x)
        x = F.normalize(x)

        return x


if __name__ == '__main__':
    model = RaMBO_backbone(512)
    input = torch.randn([2, 3, 224, 224])
    output = model(input)