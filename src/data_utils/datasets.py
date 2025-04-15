# File: ./src/data_utils/datasets.py # <<< MODIFIED >>>
# --------------------------------------------------------------------------------
# File: ./src/data_utils/datasets.py (Corrected for MNIST 50k/10k split)
import torch
import torchvision
from torch.utils.data import DataLoader, Subset, random_split, Dataset # Import Dataset
import os
import logging
from typing import Tuple, Optional, Callable

# Import transforms from the sibling module
from .preprocessing import get_transforms

logger = logging.getLogger(__name__)

# --- Custom Subset class to handle transform application ---
# <<< CORRECTION: Added TransformedSubset helper >>>
class TransformedSubset(Dataset):
    """
    A Subset that applies a specific transform independent of the original dataset's transform.
    Necessary because torchvision datasets apply transforms *before* subsetting if passed to constructor.
    """
    def __init__(self, subset: Subset, transform: Optional[Callable] = None):
        self.subset = subset
        self.transform = transform
        # Store the underlying dataset for direct access if needed
        self.dataset = subset.dataset
        logger.debug(f"Created TransformedSubset with {len(self.subset.indices)} indices. Transform applied: {self.transform is not None}")

    def __getitem__(self, index):
        # Get the *original* data item from the underlying dataset using the subset index mapping
        # We need the RAW data item before applying the transform specific to this subset
        original_index = self.subset.indices[index]
        try:
            # Access the underlying dataset directly
            data, target = self.dataset[original_index]
        except IndexError:
             logger.error(f"IndexError in TransformedSubset: index {index} maps to original_index {original_index}, but dataset size is {len(self.dataset)}")
             raise
        except Exception as e:
             logger.error(f"Error getting item {original_index} from underlying dataset: {e}")
             raise

        # Apply the specific transform provided to this subset
        if self.transform:
            data = self.transform(data)
        return data, target

    def __len__(self):
        return len(self.subset.indices)

def get_dataloaders(
    dataset_name: str,
    batch_size: int,
    data_root: str = "./data",
    val_split: float = 0.1, # <<< NOTE: This argument is ignored for MNIST when replicating reference split >>>
    seed: Optional[int] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    download: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]: # val_loader can be None
    """
    Creates training, validation, and test DataLoaders for a specified dataset.
    <<< CORRECTION: Implements the specific 50k/10k fixed split for MNIST, ignoring val_split argument >>>
    """
    dataset_name = dataset_name.lower()
    logger.info(f"Loading dataset: {dataset_name.upper()} from {data_root}")

    # --- Get Transforms ---
    try:
        train_transform = get_transforms(dataset_name, train=True)
        test_transform = get_transforms(dataset_name, train=False)
    except ValueError as e:
        logger.error(f"Failed to get transforms: {e}")
        raise

    # --- Load Datasets ---
    dataset_class = None
    if dataset_name == "fashionmnist": dataset_class = torchvision.datasets.FashionMNIST
    elif dataset_name == "mnist": dataset_class = torchvision.datasets.MNIST
    elif dataset_name == "cifar10": dataset_class = torchvision.datasets.CIFAR10
    elif dataset_name == "cifar100": dataset_class = torchvision.datasets.CIFAR100
    else: raise ValueError(f"Unknown dataset name: {dataset_name}")

    try:
        # --- Load Full Training Set (RAW - NO TRANSFORM HERE) ---
        # <<< CORRECTION: Load without transform initially to allow custom subset transforms >>>
        full_train_dataset_raw = dataset_class(
            root=data_root, train=True, download=download, transform=None,
        )
        # --- Load Test Set (With transform is fine) ---
        test_dataset = dataset_class(
            root=data_root, train=False, download=download, transform=test_transform
        )
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name.upper()}: {e}", exc_info=True)
        if isinstance(e, RuntimeError) and "download=True" in str(e) and not download:
            logger.error("Dataset not found locally and download=False.")
            raise FileNotFoundError(f"Dataset {dataset_name.upper()} not found in {data_root} and download is disabled.")
        raise

    # --- Split Train/Validation ---
    train_dataset: Optional[Dataset] = None # Use Optional since logic might fail
    val_dataset: Optional[Dataset] = None

    # <<< CORRECTION: Implement fixed 50k/10k split specifically for MNIST >>>
    if dataset_name == "mnist":
        logger.info("Applying specific MNIST fixed split: 50k train / 10k validation.")
        if len(full_train_dataset_raw) != 60000:
             logger.warning(f"MNIST training set size is not 60000 ({len(full_train_dataset_raw)}). Fixed split might be incorrect.")
             # Proceed with fixed indices anyway, but warn user.
        train_indices = list(range(50000))
        val_indices = list(range(50000, 60000))

        # Ensure indices are within bounds
        if not train_indices or not val_indices or val_indices[-1] >= len(full_train_dataset_raw):
             logger.error("Failed to apply fixed 50k/10k split to MNIST. Check dataset integrity.")
             # Indicate failure by setting datasets back to None
             train_dataset = None
             val_dataset = None
        else:
             # Apply transforms AFTER creating subsets using the helper class
             train_subset = Subset(full_train_dataset_raw, train_indices)
             val_subset = Subset(full_train_dataset_raw, val_indices)
             train_dataset = TransformedSubset(train_subset, transform=train_transform)
             val_dataset = TransformedSubset(val_subset, transform=test_transform)
             logger.info(f"Created fixed MNIST split: {len(train_dataset)} train / {len(val_dataset)} validation samples.")

    # <<< Original Random Split Logic (for other datasets or MNIST fallback) >>>
    if train_dataset is None: # Only run if fixed split failed or wasn't applicable
        logger.info(f"Using random split for dataset {dataset_name} (val_split={val_split}).")
        if not 0.0 <= val_split < 1.0:
            raise ValueError(f"Validation split must be between 0.0 and 1.0 (exclusive of 1.0), got {val_split}")

        if val_split > 0.0:
            num_train_total = len(full_train_dataset_raw)
            num_val = int(num_train_total * val_split)
            num_train_split = num_train_total - num_val
            if num_val == 0 or num_train_split == 0:
                logger.warning(f"Validation split {val_split} resulted in {num_train_split} train / {num_val} validation samples for dataset size {num_train_total}. Using full dataset for training.")
                # Use TransformedSubset even when not splitting
                train_dataset = TransformedSubset(Subset(full_train_dataset_raw, range(num_train_total)), transform=train_transform)
                val_dataset = None
            else:
                logger.info(f"Splitting training data randomly: {num_train_split} train / {num_val} validation samples.")
                generator = torch.Generator().manual_seed(seed) if seed is not None else None
                train_subset, val_subset = random_split(full_train_dataset_raw, [num_train_split, num_val], generator=generator)
                train_dataset = TransformedSubset(train_subset, transform=train_transform)
                val_dataset = TransformedSubset(val_subset, transform=test_transform)
                logger.info("Applied training transforms to train subset and test transforms to validation subset.")
        else:
            logger.info("Validation split is 0, using full training set for training.")
            train_dataset = TransformedSubset(Subset(full_train_dataset_raw, range(len(full_train_dataset_raw))), transform=train_transform)
            val_dataset = None
    # <<< END CORRECTION / Original Logic >>>

    # Check if train dataset was successfully created
    if train_dataset is None:
        raise RuntimeError("Failed to create training dataset.")

    # --- Create DataLoaders ---
    persistent_workers = num_workers > 0
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
        persistent_workers=persistent_workers,
    )
    val_loader = None
    if val_dataset and len(val_dataset) > 0:
        val_loader = DataLoader(
            dataset=val_dataset, batch_size=batch_size * 2, shuffle=False, # Larger batch size for eval usually fine
            num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
            persistent_workers=persistent_workers,
        )
    elif val_split > 0.0 and dataset_name != "mnist": # Only warn if random split was intended but failed
        logger.warning("Validation dataset is empty after random split. Validation loader will be None.")

    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
        persistent_workers=persistent_workers,
    )

    logger.info(f"DataLoaders created: Train ({len(train_loader)} batches, {len(train_dataset)} samples), Validation ({len(val_loader) if val_loader else 0} batches, {len(val_dataset) if val_dataset else 0} samples), Test ({len(test_loader)} batches, {len(test_dataset)} samples)")

    return train_loader, val_loader, test_loader

# --------------------------------------------------------------------------------