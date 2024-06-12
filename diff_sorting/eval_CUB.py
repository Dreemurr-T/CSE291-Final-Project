from model import RaMBO_backbone
from dataset import CUB200Dataset
from torch.utils.data import DataLoader
import torch
from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = RaMBO_backbone(embedding_dim=512)
model.load_state_dict(torch.load('checkpoint/best_model.pth'))
model = model.to(device)


CUB200_test_dataset = CUB200Dataset(
        root_dir='dataset/CUB_200_2011', train=False)

test_loader = DataLoader(CUB200_test_dataset, shuffle=False, num_workers=4)

def eval(model, test_loader, device, k=1):
    # Extract embeddings and labels from the test set
    embeddings, labels = extract_embeddings(model, test_loader, device)

    # Compute Recall
    recall = recall_at_k(embeddings, labels, k)

    return recall

recall_1 = eval(model, test_loader, device, 1)
recall_2 = eval(model, test_loader, device, 2)
recall_4 = eval(model, test_loader, device, 4)
recall_8 = eval(model, test_loader, device, 8)

print(recall_1, recall_2, recall_4, recall_8)