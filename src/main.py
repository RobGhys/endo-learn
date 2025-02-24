import argparse
import os
import random
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from sklearn.model_selection import train_test_split

from datasets.endoscopy_dataset import EndoscopyDataset
from models.model_manager import get_model
from trainers.trainer import EndoscopyClassificationTrainer
from transforms.base_transforms import get_transforms


def get_datasets(args):
    transforms_dict = get_transforms(model_size=(224, 224), resize_size=(256, 256))

    full_dataset = EndoscopyDataset(
        root_dir=args.data,
        transform=None
    )

    args.n_classes = len(full_dataset.classes)
    print(f"Found {args.n_classes} classes in dataset")

    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    labels = [full_dataset[i][1] for i in indices]

    train_indices, valid_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=args.seed,
        stratify=labels
    )

    train_dataset = EndoscopyDataset(
        root_dir=args.data,
        transform=transforms_dict['train'],
        indices=train_indices
    )

    valid_dataset = EndoscopyDataset(
        root_dir=args.data,
        transform=transforms_dict['valid'],
        indices=valid_indices
    )

    print(f"Training set: {len(train_indices)} images")
    print(f"Validation set: {len(valid_indices)} images")

    return train_dataset, valid_dataset

def get_data_loaders(args, datasets):
    """Create data loaders from datasets"""
    train_dataset, valid_dataset = datasets

    dataloaders = {
        'train': torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True
        ),
        'valid': torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True
        )
    }
    return dataloaders


def setup_wandb(args):
    """Initialize Weights & Biases for experiment tracking"""
    try:
        import wandb

        if args.wandb_api_key:
            os.environ["WANDB_API_KEY"] = args.wandb_api_key
        config = {
            "architecture": args.arch,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "seed": args.seed,
            "weight_decay": args.weight_decay,
            "num_classes": args.n_classes
        }

        wandb.init(
            project=f"endolearn-{args.arch}",
            config=config,
            name=f"{args.arch}_training"
        )

        return wandb
    except ImportError:
        print("Warning: wandb not installed. Running without logging.")
        return None
    except Exception as e:
        print(f"Error setting up wandb: {e}")
        return None


def get_device():
    """Determine the best available device"""
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f'Using device: {device}')
    return torch.device(device)

def get_max_workers() -> int:
    # keeps 1 for system
    return max(1, os.cpu_count() - 1)

def main():
    model_names = ['resnet18', 'resnet50', 'vgg19']

    parser = argparse.ArgumentParser(description="Endoscopy Classification Training")
    parser.add_argument("--data",
                        metavar="DIR_DATA",
                        default='/Users/rob/projects/explainable_endoscopic_vision/data/labeled-images',
                        help="path to dataset")
    parser.add_argument("--results_path", metavar="DIR_RES", help="path to save results", default="./results")
    parser.add_argument("--wandb_api_key", help="API key for Wandb")

    parser.add_argument("-a", "--arch", metavar="ARCH", default="resnet50",
                        choices=model_names, help="model architecture: " + " | ".join(model_names))
    parser.add_argument("-j", "--workers",
                        default=get_max_workers(),
                        type=int, metavar="N",
                        help="number of data loading workers")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N_EPOCH",
                        help="manual epoch number (useful on restarts)")
    parser.add_argument("--n-classes", default=None, type=int, metavar="N",
                        help="number of classes (determined automatically if not specified)")
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument("--lr", "--learning-rate", default=0.001, type=float,
                        metavar="LR", help="initial learning rate", dest="lr")
    parser.add_argument("--seed", default=42, type=int, help="seed for initialization")
    parser.add_argument('--batch-size', default=32, type=int, help='Batch size for training')
    parser.add_argument('--epochs', default=100, type=int, help='Number of total epochs to run')
    parser.add_argument('--snapshot-interval', default=10, type=int,
                        help='How often to save model snapshots')

    args = parser.parse_args()

    # Set device
    args.device = get_device()

    # Set random seeds for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably!')

    # Create results directory
    args.results_path = os.path.join(args.results_path, args.arch)
    os.makedirs(args.results_path, exist_ok=True)

    # Load datasets
    datasets = get_datasets(args)
    dataloaders = get_data_loaders(args, datasets)

    # Initialize model
    model = get_model(model_name=args.arch, num_classes=args.n_classes).to(args.device)
    print(f"Initialized {args.arch} model with {args.n_classes} output classes")

    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        [{'params': model.parameters(), 'initial_lr': args.lr}],
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Set up learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-5,
        last_epoch=-1
    )

    # Set up wandb for logging
    wandb_run = setup_wandb(args)

    # Initialize trainer
    trainer = EndoscopyClassificationTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        args=args,
        num_classes=args.n_classes,
        lr_scheduler=scheduler,
        gradient_clipping=True,
        snapshot_interval=args.snapshot_interval
    )

    # Train model
    print(f"\nStarting training for {args.epochs} epochs")
    run_summary = trainer.train(
        dataloaders['train'],
        dataloaders['valid'],
        num_epochs=args.epochs,
        start_epoch=args.start_epoch
    )

    # Finish wandb run
    if wandb_run:
        wandb_run.finish()

    print("\nTraining completed!")
    print(f"Results saved to: {args.results_path}")


if __name__ == '__main__':
    main()