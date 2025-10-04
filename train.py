import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from models.f3_histo import F3Histo
from utils.data_loader import HistopathologyDataset
from utils.metrics import compute_metrics
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def train(rank, world_size, args):
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Load dataset
    dataset = HistopathologyDataset(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        transform=transform
    )
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Initialize model
    model = F3Histo(num_classes=dataset.num_classes, num_timesteps=1000).to(device)
    model = DDP(model, device_ids=[rank])

    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion_diffusion = nn.MSELoss()
    criterion_classification = nn.CrossEntropyLoss(reduction='none')

    best_f1 = 0.0
    for epoch in range(args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        total_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            noise_pred, class_pred, agg_features = model(images)
            
            # Compute losses
            noise = torch.randn_like(images)
            loss_diffusion = criterion_diffusion(noise_pred, noise)
            
            # ACB loss
            weights = model.module.acb(agg_features, labels)
            loss_classification = criterion_classification(class_pred, labels)
            loss_classification = (loss_classification * weights).mean()
            
            loss = loss_diffusion + args.lambda_weight * loss_classification
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if rank == 0 and batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}, "
                      f"Diff Loss: {loss_diffusion.item():.4f}, Cls Loss: {loss_classification.item():.4f}")

        # Validation
        if rank == 0:
            model.eval()
            val_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    _, class_pred, _ = model(images)
                    metrics = compute_metrics(class_pred, labels)
                    for key in val_metrics:
                        val_metrics[key].append(metrics[key])

            val_metrics = {k: np.mean(v) for k, v in val_metrics.items()}
            print(f"Validation - Epoch {epoch+1}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")

            # Save best model
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                torch.save(model.module.state_dict(), os.path.join(args.checkpoint_dir, 'best_model.pth'))

    dist.destroy_process_group()

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    dataset = HistopathologyDataset(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        transform=transform
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Load model
    model = F3Histo(num_classes=dataset.num_classes, num_timesteps=1000).to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()

    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            _, class_pred, _ = model(images)
            batch_metrics = compute_metrics(class_pred, labels)
            for key in metrics:
                metrics[key].append(batch_metrics[key])

    metrics = {k: np.mean(v) for k, v in metrics.items()}
    print(f"Evaluation - Acc: {metrics['accuracy']:.4f}, Pre: {metrics['precision']:.4f}, "
          f"Rec: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")

def main():
    parser = argparse.ArgumentParser(description="F3-Histo Training and Evaluation")
    parser.add_argument('--dataset', type=str, choices=['BreakHis', 'NCT-CRC-HE-100K', 'GasHisSDB-160', 'ROSE'],
                        default='BreakHis', help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='data/BreakHis', help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lambda_weight', type=float, default=0.5, help='Classification loss weight')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth', help='Path to checkpoint for evaluation')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation mode')
    
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    world_size = torch.cuda.device_count()
    if args.evaluate:
        evaluate(args)
    else:
        mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()