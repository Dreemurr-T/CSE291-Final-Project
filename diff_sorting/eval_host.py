from model import RaMBO_backbone
from dataset import CUB200Dataset
from torch.utils.data import DataLoader
import torch
from utils import *

def load_model(dataset):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = RaMBO_backbone(embedding_dim=512)

    model.load_state_dict(torch.load(f'checkpoint/{dataset}/best_model.pth'))
    model = model.to(device)

    return model, device

def eval(dataset, model, device):
    if dataset == "CUB200":
        CUB200_test_dataset = CUB200Dataset(
            root_dir='dataset/CUB_200_2011', train=False)

    test_loader = DataLoader(CUB200_test_dataset, shuffle=False, num_workers=4)

    embeddings, labels = extract_embeddings(model, test_loader, device)

    recall_list = []

    # Compute Recall
    if dataset == "CUB200":
        recall_list.append(recall_at_k(embeddings, labels, k) for k in [1, 2, 4, 8])
    elif dataset == "SOP":
        recall_list.append(recall_at_k(embeddings, labels, k) for k in [1, 10, 100, 1000])

    return recall_list


if __name__ == '__main__':
    dataset = "CUB200"          # change dataset here for different evaluation settings
    model, device = load_model(dataset)
    recalls = eval(dataset, model, device)

    print(recalls)