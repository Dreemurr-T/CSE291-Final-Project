import torch
import numpy as np
import yaml
import faiss


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


# Evaluate Recall@k using Faiss for efficient nearest neighbor search
def evaluate_recall_at_k_faiss(model, dataloader, device, k=1):
    model.eval()
    embeddings, labels = extract_embeddings(model, dataloader, device)

    # Convert embeddings to float32 for Faiss
    embeddings = embeddings.astype(np.float32)
    
    # Build the Faiss index
    index = faiss.IndexFlatL2(embeddings.shape[1])  # Use L2 distance
    index.add(embeddings)  # Add embeddings to the index

    # Perform search
    D, I = index.search(embeddings, k+1)  # Search for the 2 nearest neighbors (includes the point itself)

    # Compute Recall@k
    recall_at_k = 0
    for i in range(len(labels)):
        if labels[i] in labels[I[i][1:k+1]]:  # Check if the closest neighbor (excluding itself) has the same label
            recall_at_k += 1
    recall_at_k /= len(labels)
    
    return recall_at_k


def lr_lambda(epoch):
    if epoch < 35:
        return 1.0
    else:
        return 0.3  # Drop learning rate to 30% of the original
