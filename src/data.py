# src/data.py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloader(dataset_name="cifar10", batch_size=64):
    """Create and return a DataLoader for the specified dataset"""
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    if dataset_name.lower() == "cifar10":
        dataset = datasets.CIFAR10(
            root="./data",
            download=True,
            transform=transform
        )
    elif dataset_name.lower() == "mnist":
        transform.transforms[-1] = transforms.Normalize((0.5,), (0.5,))  # Single channel
        dataset = datasets.MNIST(
            root="./data",
            download=True,
            transform=transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )