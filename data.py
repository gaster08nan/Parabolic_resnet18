import torch
import torchvision
import torchvision.transforms as transforms


# CIFAR-10 normalization statistics
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def get_transforms(train=True):
    """Get data transforms for training or testing.
    
    Training includes data augmentation:
    - Random horizontal flip
    - Random crop with padding
    
    Testing only applies normalization.
    """
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])


def load_cifar10(batch_size=128, test_batch_size=100, num_workers=2):
    """Load CIFAR-10 dataset with DataLoaders.
    
    Args:
        batch_size: Batch size for training
        test_batch_size: Batch size for testing
        num_workers: Number of worker processes for data loading
        
    Returns:
        trainloader, testloader, classes
    """
    transform_train = get_transforms(train=True)
    transform_test = get_transforms(train=False)
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=test_batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, testloader, classes
