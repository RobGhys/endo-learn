import os
import random
import torch
from torch.utils.data import Dataset, random_split
from PIL import Image
from torchvision import transforms
from typing import Optional, Tuple, List, Dict
import matplotlib.pyplot as plt
import numpy as np


class EndoscopyDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 transform: Optional[transforms.Compose] = None,
                 indices=None):
        self.root_dir = root_dir
        self.transform = transform
        self.indices = indices

        self.filter_path = root_dir

        self.classes = []
        self.class_to_idx = {}
        self.imgs = []

        self._find_leaf_classes_and_images()

    def _find_leaf_classes_and_images(self):
        leaf_folders = []

        for dirpath, dirnames, filenames in os.walk(self.filter_path):
            has_images = any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in filenames)

            has_image_subdirs = False
            for d in dirnames:
                subdir = os.path.join(dirpath, d)
                if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in os.listdir(subdir)):
                    has_image_subdirs = True
                    break

            if has_images and not has_image_subdirs:
                leaf_folders.append(dirpath)

        leaf_folders.sort()

        for idx, folder in enumerate(leaf_folders):
            class_name = os.path.basename(folder)
            self.classes.append(class_name)
            self.class_to_idx[class_name] = idx

            for img_name in os.listdir(folder):
                if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue

                img_path = os.path.join(folder, img_name)
                self.imgs.append((img_path, idx))

    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        return len(self.imgs)

    def __getitem__(self, idx):
        if self.indices is not None:
            idx = self.indices[idx]

        img_path, class_idx = self.imgs[idx]

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, class_idx

    def get_class_distribution(self):
        distribution = {cls: 0 for cls in self.classes}

        for _, class_idx in self.imgs:
            class_name = self.classes[class_idx]
            distribution[class_name] += 1

        return distribution

    def get_sample_images(self, num_per_class=3):
        """Get sample images from each class for visualization"""
        samples = {}

        for class_name in self.classes:
            class_idx = self.class_to_idx[class_name]
            class_images = [img_path for img_path, idx in self.imgs if idx == class_idx]

            if class_images:
                # Select random samples (or all if less than requested)
                num_samples = min(num_per_class, len(class_images))
                selected = random.sample(class_images, num_samples)
                samples[class_name] = selected

        return samples


def visualize_dataset_info(dataset, save_path=None):
    """Visualize dataset information and statistics"""
    # Get class distribution
    distribution = dataset.get_class_distribution()

    # Basic statistics
    total_images = len(dataset)
    num_classes = len(dataset.classes)

    print(f"Dataset Statistics:")
    print(f"Total images: {total_images}")
    print(f"Number of classes: {num_classes}")
    print("\nClass distribution:")

    # Sort classes by count for better visualization
    sorted_dist = {k: v for k, v in sorted(distribution.items(), key=lambda item: item[1], reverse=True)}

    for cls, count in sorted_dist.items():
        print(f"  {cls}: {count} images ({count / total_images * 100:.1f}%)")

    # Plot class distribution
    plt.figure(figsize=(12, 6))
    plt.bar(sorted_dist.keys(), sorted_dist.values())
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.title('Class Distribution')
    plt.xticks(rotation=90)
    plt.tight_layout()

    if save_path:
        plt.savefig(os.path.join(save_path, 'class_distribution.png'))
    else:
        plt.show()

    # Get and display sample images
    samples = dataset.get_sample_images(num_per_class=3)

    fig, axes = plt.subplots(len(samples), 3, figsize=(12, 3 * len(samples)))

    # Handle case with only one class
    if len(samples) == 1:
        axes = np.array([axes])

    for i, (class_name, image_paths) in enumerate(samples.items()):
        for j, img_path in enumerate(image_paths):
            if j < 3:  # Just in case we have less than 3 images for some classes
                img = Image.open(img_path).convert('RGB')
                axes[i, j].imshow(img)
                axes[i, j].set_title(f"{class_name}")
                axes[i, j].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, 'sample_images.png'))
    else:
        plt.show()


if __name__ == '__main__':
    import argparse
    from transforms.base_transforms import get_transforms

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize Endoscopy Dataset')
    parser.add_argument('--data', type=str, default='/Users/rob/projects/explainable_endoscopic_vision/data/labeled-images',
                        help='Path to dataset root')
    parser.add_argument('--output', type=str, default=None, help='Path to save visualizations')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    # Initialize dataset with transforms
    transforms_dict = get_transforms()
    dataset = EndoscopyDataset(root_dir=args.data, transform=transforms_dict['valid'])

    # Print dataset information
    print(f"\nAnalyzing dataset at: {args.data}")
    visualize_dataset_info(dataset, save_path=args.output)

    # Split dataset and print split statistics
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size

    train_dataset, valid_dataset = random_split(
        dataset, [train_size, valid_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"\nDataset splits:")
    print(f"  Training set: {len(train_dataset)} images ({train_size / len(dataset) * 100:.1f}%)")
    print(f"  Validation set: {len(valid_dataset)} images ({valid_size / len(dataset) * 100:.1f}%)")