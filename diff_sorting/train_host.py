import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import ctypes
import compiler

import time
import torch.optim as optim
from model import RaMBO_backbone
from dataset import CUB200Dataset, SOPDataset
from torch.utils.data import DataLoader
import torch
from utils import *


with open('loma_code/diff_RaMBO_simplified.py') as f:
    _, lib = compiler.compile(f.read(),
                              target='c',
                              output_filename='_code/diff_RaMBO_simplified')


def trainer(epoch, train_dataloader, device, model, config, optimizer):
    for iter, train_data in enumerate(train_dataloader):
        optimizer.zero_grad()
        start_time = time.time()

        images, labels = train_data[0], train_data[1]
        images = images.to(device)

        features = model(images)
        scores = features @ features.t()        # Get cosine similarity
        score_gradients = torch.zeros_like(scores)
        loss_record = ctypes.c_float(0.0)

        for i, score_tensor in enumerate(scores):
            py_score = score_tensor.detach().cpu().numpy().tolist()
            py_label = labels[i].cpu().numpy().tolist()

            py_dscore = [0.0] * score_tensor.shape[0]
            py_rank = [0] * score_tensor.shape[0]
            py_norm_rank = [0.0] * score_tensor.shape[0]
            py_d_norm_rank = [0.0] * score_tensor.shape[0]

            d_score = (ctypes.c_float * len(py_dscore))(*py_dscore)
            rank = (ctypes.c_int * len(py_rank))(*py_rank)
            norm_rank = (ctypes.c_float * len(py_norm_rank))(*py_norm_rank)
            d_norm_rank = (ctypes.c_float * len(py_norm_rank))(*py_d_norm_rank)
            score = (ctypes.c_float * len(py_score))(*py_score)
            label = (ctypes.c_int * len(py_label))(*py_label)

            size = ctypes.c_int(len(py_score))
            lambda_val = ctypes.c_float(config['lambda'])

            lib.call_RaMBO(score, d_score, size, rank, norm_rank,
                           d_norm_rank, label, lambda_val, loss_record)

            score_gradient = torch.tensor(list(d_score))
            score_gradients[i] = score_gradient

        scores.backward(score_gradients)
        optimizer.step()

        end_time = time.time()

        print("Epoch: %d / %d Iteration: %d / %d Elapsed time: %fs Loss: %f" % (
            epoch+1, config['epoch'], iter+1, len(train_dataloader), end_time-start_time, loss_record.value / scores.shape[0]))


def train(config):
    if config['dataset'] == "CUB200":
        train_dataset = CUB200Dataset(
            root_dir='dataset/CUB_200_2011', train=True)
        test_dataset = CUB200Dataset(
            root_dir='dataset/CUB_200_2011', train=False)
        
    elif config['dataset'] == "SOP":
        train_dataset = SOPDataset(
            root_dir='dataset/Stanford_Online_Products', train=True)
        test_dataset = SOPDataset(
            root_dir='dataset/Stanford_Online_Products', train=False)
    
    train_dataloader = DataLoader(
            train_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=8, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, shuffle=False, num_workers=8)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = RaMBO_backbone(embedding_dim=512).to(device)

    optimizer = optim.Adam([
        {'params': model.resnet50.conv1.parameters()},
        {'params': model.resnet50.bn1.parameters()},
        {'params': model.resnet50.layer1.parameters()},
        {'params': model.resnet50.layer2.parameters()},
        {'params': model.resnet50.layer3.parameters()},
        {'params': model.resnet50.layer4.parameters()},
        {'params': model.resnet50.fc.parameters(
        ), 'lr': config['learning_rate'] * 10}
    ], lr=config['learning_rate'], weight_decay=config['weight_decay'])

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if not os.path.exists(f"checkpoint/{config['dataset']}"):
        os.makedirs(f"checkpoint/{config['dataset']}")

    record_file = f"{config['dataset']}_{config['batch_size']}.txt"

    best_recall = 0.0

    for epoch in range(config['epoch']):
        model = model.train()
        trainer(epoch, train_dataloader, device, model, config, optimizer)

        scheduler.step()

        save_path = os.path.join(f"checkpoint/{config['dataset']}", f"{config['dataset']}_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)

        best_save_path = os.path.join(f"checkpoint/{config['dataset']}", "best_model.pth")
        recall = evaluate_recall_at_k_faiss(model, test_dataloader, device, k=1)

        with open(record_file, 'a') as file:
            file.write(f"Epoch {epoch+1}: Recall@1 = {recall:.4f}\n")

        if recall >= best_recall:
            best_recall = recall
            torch.save(model.state_dict(), best_save_path)

        print(f"Epoch: {epoch+1} Recall@1: {recall:.4f}, Best Recall@1: {best_recall:.4f}")


# def eval(model, test_loader, device, k=1):
#     # Extract embeddings and labels from the test set
#     embeddings, labels = extract_embeddings(model, test_loader, device)

#     # Compute Recall@1
#     recall_at_1 = recall_at_k(embeddings, labels, k)

#     return recall_at_1


if __name__ == '__main__':
    config = get_config('scripts/RaMBO.yaml')
    print("Training configs for CUB200 dataset:", config)
    train(config)