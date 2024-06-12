import torch
import numpy as np
import yaml
from sklearn.metrics.pairwise import cosine_distances


def get_config(config_path):
    config = {}

    with open(config_path, 'r') as file:
        yaml_config = yaml.safe_load(file)
        config = yaml_config

    return config

def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels)

    # Create the label matrix
    label_matrix = torch.zeros((len(labels), len(labels)), dtype=torch.int)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if labels[i] == labels[j]:
                label_matrix[i, j] = 1

    return images, label_matrix


def extract_embeddings(model, dataloader, device):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for images, lbls in dataloader:
            images = images.to(device)
            outputs = model(images)
            embeddings.append(outputs.cpu().numpy())
            labels.append(lbls.cpu().numpy())
    embeddings = np.vstack(embeddings)
    labels = np.hstack(labels)
    return embeddings, labels

def recall_at_k(embeddings, labels, k=1):
    distances = cosine_distances(embeddings)
    np.fill_diagonal(distances, np.inf)
    nearest_neighbors = np.argpartition(distances, k, axis=1)[:, :k]
    recalls = 0
    for i in range(len(labels)):
        if labels[i] in labels[nearest_neighbors[i]]:
            recalls += 1
    return recalls / len(labels)


def lr_lambda(epoch):
    if epoch < 35:
        return 1.0
    else:
        return 0.3  # Drop learning rate by 70%