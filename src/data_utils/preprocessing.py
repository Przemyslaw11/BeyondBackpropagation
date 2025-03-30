import torch
import torchvision.transforms as transforms
from typing import Dict, Tuple

# Define normalization constants for each dataset
# Values typically derived from the dataset's training split
# Using standard values often found online or in torchvision examples

# Fashion-MNIST (Grayscale, 1 channel)
# Mean and std calculated from training set: (0.2860,), (0.3530,) approx.
# Often simplified to (0.5,), (0.5,) for normalization to [-1, 1] range
FASHION_MNIST_MEAN = (0.2860,)
FASHION_MNIST_STD = (0.3530,)
# Alternative: FASHION_MNIST_MEAN = (0.5,)
# Alternative: FASHION_MNIST_STD = (0.5,)


# CIFAR-10 (RGB, 3 channels)
# Values from torchvision documentation/common practice
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010) # Often approximated as (0.247, 0.243, 0.261) based on other sources

# CIFAR-100 (RGB, 3 channels)
# Values often similar to CIFAR-10, or calculated specifically
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def get_transforms(dataset_name: str, train: bool = True) -> transforms.Compose:
    """
    Returns the appropriate torchvision transforms for a given dataset and split.

    Args:
        dataset_name: Name of the dataset ('FashionMNIST', 'CIFAR10', 'CIFAR100').
        train: Boolean indicating if the transforms are for the training set (True)
               or validation/test set (False). Augmentation is applied only if train is True.

    Returns:
        A torchvision.transforms.Compose object.

    Raises:
        ValueError: If the dataset_name is not recognized.
    """
    dataset_name = dataset_name.lower()

    if dataset_name == 'fashionmnist':
        mean = FASHION_MNIST_MEAN
        std = FASHION_MNIST_STD
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
        # No standard augmentation for FashionMNIST in typical benchmarks
        # Add if needed based on specific experiments

    elif dataset_name in ['cifar10', 'cifar100']:
        if dataset_name == 'cifar10':
            mean = CIFAR10_MEAN
            std = CIFAR10_STD
        else: # cifar100
            mean = CIFAR100_MEAN
            std = CIFAR100_STD

        if train:
            # Standard CIFAR augmentation
            transform_list = [
                transforms.RandomCrop(32, padding=4), # Pad and then randomly crop back to 32x32
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]
        else:
            # No augmentation for validation/test
            transform_list = [
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}. Choose from 'FashionMNIST', 'CIFAR10', 'CIFAR100'.")

    return transforms.Compose(transform_list)


if __name__ == '__main__':
    print("Testing preprocessing utils...")

    # Test getting transforms for each dataset and split
    try:
        print("\n--- FashionMNIST ---")
        train_transform_fmnist = get_transforms('FashionMNIST', train=True)
        test_transform_fmnist = get_transforms('FashionMNIST', train=False)
        print("Train Transform (FMNIST):", train_transform_fmnist)
        print("Test Transform (FMNIST):", test_transform_fmnist)
        # Check if they are different (they shouldn't be for FMNIST in this setup)
        assert str(train_transform_fmnist) == str(test_transform_fmnist)

        print("\n--- CIFAR10 ---")
        train_transform_cifar10 = get_transforms('CIFAR10', train=True)
        test_transform_cifar10 = get_transforms('CIFAR10', train=False)
        print("Train Transform (CIFAR10):", train_transform_cifar10)
        print("Test Transform (CIFAR10):", test_transform_cifar10)
        # Check if they are different (they should be due to augmentation)
        assert str(train_transform_cifar10) != str(test_transform_cifar10)

        print("\n--- CIFAR100 ---")
        train_transform_cifar100 = get_transforms('CIFAR100', train=True)
        test_transform_cifar100 = get_transforms('CIFAR100', train=False)
        print("Train Transform (CIFAR100):", train_transform_cifar100)
        print("Test Transform (CIFAR100):", test_transform_cifar100)
        assert str(train_transform_cifar100) != str(test_transform_cifar100)

        # Test applying transforms (requires dummy data)
        print("\n--- Applying Transforms (Example) ---")
        # Dummy CIFAR-like image (3x32x32)
        dummy_cifar_img = torch.rand(3, 32, 32)
        transformed_train = train_transform_cifar10(transforms.ToPILImage()(dummy_cifar_img)) # Need PIL for augmentations
        transformed_test = test_transform_cifar10(dummy_cifar_img) # ToTensor handles tensor input directly
        print("Original shape:", dummy_cifar_img.shape)
        print("Transformed (Train) shape:", transformed_train.shape)
        print("Transformed (Test) shape:", transformed_test.shape)
        assert transformed_train.shape == dummy_cifar_img.shape
        assert transformed_test.shape == dummy_cifar_img.shape
        # Values should be normalized (mean approx 0, std approx 1)
        print(f"Transformed (Test) mean: {transformed_test.mean():.4f}, std: {transformed_test.std():.4f}")


        # Test invalid dataset name
        print("\n--- Testing Invalid Dataset ---")
        try:
            get_transforms('InvalidDataset')
        except ValueError as e:
            print(f"Caught expected error: {e}")

    except Exception as e:
        print(f"\nAn error occurred during testing: {e}")
