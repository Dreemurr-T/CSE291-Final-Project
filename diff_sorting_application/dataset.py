import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


# Define the transform
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to 256x256
    transforms.RandomCrop((224, 224)),  # Random crop to 224x224
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.ToTensor(),  # Convert the image to a tensor
    # Normalize using ImageNet mean and std
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to 256x256
    transforms.CenterCrop((224, 224)),  # Random crop to 224x224
    transforms.ToTensor(),  # Convert the image to a tensor
    # Normalize using ImageNet mean and std
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class CUB200Dataset(Dataset):
    def __init__(self, root_dir, train=True):
        self.root_dir = root_dir
        self.train = train
        if self.train:
            self.transform = train_transform
        else:
            self.transform = test_transform

        # Load the image paths and labels
        self.image_paths, self.labels = self._load_metadata()
        self.class_to_indices = self._group_by_class()

    def _load_metadata(self):
        images_file = os.path.join(self.root_dir, 'images.txt')
        labels_file = os.path.join(self.root_dir, 'image_class_labels.txt')
        train_test_file = os.path.join(self.root_dir, 'train_test_split.txt')

        # Load data from files
        images = pd.read_csv(images_file, sep=' ', header=None, names=[
                             'img_id', 'filepath'])
        labels = pd.read_csv(labels_file, sep=' ',
                             header=None, names=['img_id', 'label'])
        train_test_split = pd.read_csv(
            train_test_file, sep=' ', header=None, names=['img_id', 'is_train'])

        # Merge dataframes on img_id
        data = images.merge(labels, on='img_id')
        data = data.merge(train_test_split, on='img_id')

        if self.train:
            data = data[data['is_train'] == 1]
        else:
            data = data[data['is_train'] == 0]

        image_paths = data['filepath'].values
        labels = data['label'].values - 1  # Labels should start from 0

        return image_paths, labels
    
    def _group_by_class(self):
        class_to_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in class_to_indices:
                class_to_indices[label] = []
            class_to_indices[label].append(idx)
        return class_to_indices

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, 'images', self.image_paths[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class SOPDataset(Dataset):
    def __init__(self, root_dir, train=True):
        self.root_dir = root_dir
        self.train = train
        if self.train:
            self.transform = train_transform
        else:
            self.transform = test_transform
        self.image_paths, self.labels = self._load_metadata()

    def _load_metadata(self):
        if self.train:
            metadata_file = os.path.join(self.root_dir, 'Ebay_train.txt')
        else:
            metadata_file = os.path.join(self.root_dir, 'Ebay_test.txt')

        # Read the metadata file
        data = pd.read_csv(metadata_file, sep=' ', header=0)

        # Extract image paths and labels
        image_paths = data['path'].values
        labels = data['class_id'].values - 1  # Labels should start from 0

        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_paths[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
