from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader, random_split

def load_wikiart_datasets(data_path='wikiart_subset', batch_size=128):
    """Create data loaders for the WikiArt dataset organized by folders.

    Assumes the dataset is organized in subdirectories within the `data_path` folder,
    with each subdirectory named after a class label (e.g., styles or genres).

    Returns: Dict containing data loaders.
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Starting with ImageNet's mean and std
                                     std=[0.229, 0.224, 0.225])  # Can be adjusted if necessary

    # Define transformations for training, validation, and testing
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),  # Resize and crop to 224x224 pixels
        transforms.RandomHorizontalFlip(),  # Data augmentation
        transforms.ToTensor(),
        normalize])

    test_transform = transforms.Compose([
        transforms.Resize(256),  # Resize the shorter side to 256 before cropping
        transforms.CenterCrop(224),  # Center crop the image to 224x224 pixels
        transforms.ToTensor(),
        normalize])

    # Load the entire dataset using ImageFolder
    full_dataset = datasets.ImageFolder(root=data_path, transform=train_transform)

    # Define sizes for train_subset, valid_subset, train_dataset, and test_dataset
    total_size = len(full_dataset)
    train_subset_size = int(0.8 * total_size)
    valid_subset_size = int(0.1 * total_size)
    train_dataset_size = train_subset_size + valid_subset_size
    test_dataset_size = total_size - train_dataset_size

    # Split the dataset into train_subset, valid_subset, train_dataset, and test_dataset
    train_subset, valid_subset, test_dataset = random_split(
        full_dataset, [train_subset_size, valid_subset_size, test_dataset_size])

    # Apply the test transform to the validation and test datasets
    valid_subset.dataset.transform = test_transform
    test_dataset.dataset.transform = test_transform

    # Create data loaders
    data_loaders = {}
    data_loaders['train_subset'] = DataLoader(dataset=train_subset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              pin_memory=True,
                                              num_workers=4)

    data_loaders['valid_subset'] = DataLoader(dataset=valid_subset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=4)

    data_loaders['train_dataset'] = DataLoader(dataset=train_subset.dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=4)

    data_loaders['test_dataset'] = DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=4)

    return data_loaders
