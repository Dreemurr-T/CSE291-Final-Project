import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import random

class SOPDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = self._construct_data_dict()

    def _construct_data_dict(self):
        data = {}
        # Assume the dataset file is structured as: image_path, class_id, super_label
        with open(os.path.join(self.root_dir, 'Ebay_train.txt'), 'r') as f:
            lines = f.readlines()[1:]  # Skip the header
            for line in lines:
                parts = line.strip().split()
                img_path = os.path.join(self.root_dir, parts[3])
                class_id = int(parts[1])
                super_label = int(parts[2])
                sample = (img_path, class_id)
                if super_label not in data:
                    data[super_label] = []
                data[super_label].append(sample)
        return data

    def __len__(self):
        return sum(len(samples) for samples in self.data.values())

    def __getitem__(self, idx):
        # This method won't be directly used as we will prepare batches manually
        pass

    def get_samples(self, super_label, num_samples):
        samples = random.sample(self.data[super_label], num_samples)
        images, labels = zip(*[(self._load_image(img_path), label) for img_path, label in samples])
        return images, labels

    def _load_image(self, img_path):
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

class SuperLabelBatchSampler:
    def __init__(self, dataset, num_batches_per_pair=10, samples_per_class=4):
        self.dataset = dataset
        self.super_labels = list(dataset.data.keys())
        self.num_batches_per_pair = num_batches_per_pair
        self.samples_per_class = samples_per_class

    def __iter__(self):
        while True:
            # Randomly select a pair of super-labels
            super_label_pair = random.sample(self.super_labels, 2)
            for _ in range(self.num_batches_per_pair):
                batch_images = []
                batch_labels = []
                for super_label in super_label_pair:
                    images, labels = self.dataset.get_samples(super_label, self.samples_per_class)
                    batch_images.extend(images)
                    batch_labels.extend(labels)
                yield batch_images, batch_labels

    def __len__(self):
        return 1000000  # Infinite iterator, arbitrary large number

def collate_fn(batch):
    batch_images, batch_labels = zip(*batch)
    images = torch.stack([item for sublist in batch_images for item in sublist])
    labels = torch.tensor([item for sublist in batch_labels for item in sublist])
    return images, labels

if __name__ == "__main__":
    # Define the root directory of the SOP dataset
    root_dir = 'dataset/Stanford_Online_Products'

    # Define the transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Instantiate the dataset
    dataset = SOPDataset(root_dir=root_dir, transform=transform)

    # Instantiate the batch sampler
    batch_sampler = SuperLabelBatchSampler(dataset)

    # Instantiate the DataLoader
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, batch_size=1)

    # Get a single batch
    for images, labels in dataloader:
        print("Images shape:", images.shape)
        print("Labels shape:", labels.shape)
        break  # Just to demonstrate