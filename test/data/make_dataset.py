import torch
import os

if __name__ == "__main__":
    # Get the data and process it

    base_path = "data/raw/corruptmnist/"

    train_images_list = []
    train_targets_list = []

    for i in range(10):
        train_images_list.append(torch.load(os.path.join(base_path, f"train_images_{i}.pt")))
        train_targets_list.append(torch.load(os.path.join(base_path, f"train_target_{i}.pt")))

    # Concatenate all train data
    train_images = torch.cat(train_images_list, dim=0)
    train_targets = torch.cat(train_targets_list, dim=0)

    mean_train = train_images.mean()
    std_train = train_images.std()
    train_images = (train_images - mean_train) / (std_train + 1e-8)  # Normalize train data

    # Test data
    test_images = torch.load(os.path.join(base_path, "test_images.pt"))
    test_images = (test_images - mean_train) / (std_train + 1e-8)  # Normalize test data

    test_targets = torch.load(os.path.join(base_path, "test_target.pt"))

    # Save normalized data to data/processed/corruptmnist/
    torch.save(test_images, "data/processed/corruptmnist/test_images.pt")
    torch.save(test_targets, "data/processed/corruptmnist/test_target.pt")
    torch.save(train_images, "data/processed/corruptmnist/train_images.pt")
    torch.save(train_targets, "data/processed/corruptmnist/train_target.pt")
