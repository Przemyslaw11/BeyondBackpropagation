import torch
import torchvision
from torch.utils.data import DataLoader, Subset, random_split
import os
import logging
from typing import Tuple, Optional

# Import transforms from the sibling module
from .preprocessing import get_transforms

logger = logging.getLogger(__name__)

def get_dataloaders(
    dataset_name: str,
    batch_size: int,
    data_root: str = './data',
    val_split: float = 0.1,
    seed: Optional[int] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    download: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates training, validation, and test DataLoaders for a specified dataset.

    Args:
        dataset_name: Name of the dataset ('FashionMNIST', 'CIFAR10', 'CIFAR100').
        batch_size: Number of samples per batch.
        data_root: Root directory where the dataset will be stored/downloaded.
        val_split: Fraction of the training data to use for validation (e.g., 0.1 for 10%).
        seed: Optional random seed for the train/validation split for reproducibility.
        num_workers: Number of subprocesses to use for data loading.
        pin_memory: If True, the data loader will copy Tensors into CUDA pinned memory
                    before returning them (recommended for GPU training).
        download: If True, downloads the dataset if it's not found in data_root.

    Returns:
        A tuple containing (train_loader, val_loader, test_loader).

    Raises:
        ValueError: If the dataset_name is not recognized or val_split is invalid.
        FileNotFoundError: If download is False and the dataset is not found.
    """
    dataset_name = dataset_name.lower()
    logger.info(f"Loading dataset: {dataset_name.upper()} from {data_root}")

    if not 0.0 <= val_split < 1.0:
        raise ValueError(f"Validation split must be between 0.0 and 1.0 (exclusive of 1.0), got {val_split}")

    # --- Get Transforms ---
    try:
        train_transform = get_transforms(dataset_name, train=True)
        test_transform = get_transforms(dataset_name, train=False) # Same transform for val and test
    except ValueError as e:
        logger.error(f"Failed to get transforms: {e}")
        raise

    # --- Load Datasets ---
    dataset_class = None
    if dataset_name == 'fashionmnist':
        dataset_class = torchvision.datasets.FashionMNIST
    elif dataset_name == 'cifar10':
        dataset_class = torchvision.datasets.CIFAR10
    elif dataset_name == 'cifar100':
        dataset_class = torchvision.datasets.CIFAR100
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    try:
        # Load raw training data (will be split)
        full_train_dataset = dataset_class(
            root=data_root,
            train=True,
            download=download,
            transform=train_transform # Apply train transforms initially
        )

        # Load test data
        test_dataset = dataset_class(
            root=data_root,
            train=False,
            download=download,
            transform=test_transform
        )
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name.upper()}: {e}", exc_info=True)
        if isinstance(e, RuntimeError) and "download=True" in str(e) and not download:
             logger.error("Dataset not found locally and download=False.")
             raise FileNotFoundError(f"Dataset {dataset_name.upper()} not found in {data_root} and download is disabled.")
        raise # Re-raise other errors

    # --- Split Train/Validation ---
    num_train = len(full_train_dataset)
    num_val = int(num_train * val_split)
    num_train_split = num_train - num_val

    logger.info(f"Splitting training data: {num_train_split} train / {num_val} validation samples.")

    if num_val == 0 and val_split > 0:
         logger.warning(f"Validation split resulted in 0 samples (val_split={val_split}, total_train={num_train}). Validation loader will be empty.")
         train_dataset = full_train_dataset
         val_dataset = None # Or an empty dataset
    elif num_val == 0 and val_split == 0:
         logger.info("Validation split is 0, using full training set for training.")
         train_dataset = full_train_dataset
         val_dataset = None
    else:
        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        train_subset_indices, val_subset_indices = random_split(
            range(num_train), [num_train_split, num_val], generator=generator
        )

        # Create Subset objects. IMPORTANT: Validation set should use test transforms!
        train_dataset = Subset(full_train_dataset, train_subset_indices)

        # Create a new dataset instance for validation with test transforms
        # This is safer than trying to change transforms on the fly for a subset
        val_dataset_raw = dataset_class(
             root=data_root,
             train=True, # Still loading from the original training data pool
             download=False, # Should already be downloaded
             transform=test_transform # Apply TEST transforms to validation data
        )
        val_dataset = Subset(val_dataset_raw, val_subset_indices)


    # --- Create DataLoaders ---
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True # Often beneficial for training stability, esp. with BatchNorm
    )

    # Handle case where val_dataset might be None or empty
    if val_dataset and len(val_dataset) > 0:
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size, # Often use same or larger batch size for validation
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
    else:
        logger.warning("Validation dataset is empty or None. Validation loader will be None.")
        val_loader = None


    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size, # Often use same or larger batch size for testing
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    logger.info(f"DataLoaders created: Train ({len(train_loader)} batches), Validation ({len(val_loader) if val_loader else 0} batches), Test ({len(test_loader)} batches)")

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    print("Testing dataset utils...")

    # Common settings
    test_batch_size = 64
    test_data_root = './data_test' # Use a separate directory for testing downloads
    test_seed = 42

    # Clean up previous test data if necessary
    # import shutil
    # if os.path.exists(test_data_root):
    #     shutil.rmtree(test_data_root)

    datasets_to_test = ['FashionMNIST', 'CIFAR10', 'CIFAR100']

    for ds_name in datasets_to_test:
        print(f"\n--- Testing {ds_name} ---")
        try:
            train_dl, val_dl, test_dl = get_dataloaders(
                dataset_name=ds_name,
                batch_size=test_batch_size,
                data_root=test_data_root,
                val_split=0.1,
                seed=test_seed,
                num_workers=0, # Use 0 workers for easier debugging in main thread
                pin_memory=False, # Often causes issues if CUDA not fully configured
                download=True
            )

            print(f"DataLoaders obtained for {ds_name}.")

            # Check a batch from each loader
            print("Checking train_loader batch...")
            train_batch_images, train_batch_labels = next(iter(train_dl))
            print(f"  Train Batch - Images shape: {train_batch_images.shape}, Labels shape: {train_batch_labels.shape}")
            assert train_batch_images.shape[0] == test_batch_size
            assert train_batch_labels.shape[0] == test_batch_size

            if val_dl:
                print("Checking val_loader batch...")
                val_batch_images, val_batch_labels = next(iter(val_dl))
                print(f"  Val Batch - Images shape: {val_batch_images.shape}, Labels shape: {val_batch_labels.shape}")
                # Validation batch size might be smaller if not dropping last and dataset size isn't multiple
                assert val_batch_images.shape[0] <= test_batch_size
                assert val_batch_labels.shape[0] <= test_batch_size
            else:
                print("Validation loader is None (as expected if split is 0 or too small).")


            print("Checking test_loader batch...")
            test_batch_images, test_batch_labels = next(iter(test_dl))
            print(f"  Test Batch - Images shape: {test_batch_images.shape}, Labels shape: {test_batch_labels.shape}")
            assert test_batch_images.shape[0] <= test_batch_size
            assert test_batch_labels.shape[0] <= test_batch_size

        except (ValueError, FileNotFoundError, Exception) as e:
            print(f"Error testing {ds_name}: {e}")
            # Depending on the environment (e.g., no internet), download might fail.

    print("\nTesting edge case: val_split = 0")
    try:
        train_dl_no_val, val_dl_no_val, _ = get_dataloaders(
            dataset_name='FashionMNIST', batch_size=test_batch_size, data_root=test_data_root,
            val_split=0.0, seed=test_seed, num_workers=0, pin_memory=False, download=True
        )
        assert val_dl_no_val is None
        print("val_split=0 handled correctly (val_loader is None).")
        # Check train loader size (should be larger than with 10% split)
        print(f"Train batches with val_split=0: {len(train_dl_no_val)}")
    except Exception as e:
        print(f"Error testing val_split=0: {e}")

    print("\nDataset utils testing finished.")
    # Consider cleaning up test_data_root afterwards if desired
    # if os.path.exists(test_data_root):
    #     shutil.rmtree(test_data_root)
    #     print(f"Cleaned up {test_data_root}")
