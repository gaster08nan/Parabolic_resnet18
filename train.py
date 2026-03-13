"""
ResNet18 Training Script for CIFAR-10

This script trains ResNet18 on CIFAR-10 using two optimization methods:
1. Standard SGD with momentum (normal loss)
2. Parabolic approximation optimizer

Comparison results are saved and plotted.
"""

import torch
import argparse
from datetime import datetime

from model import create_model
from data import load_cifar10
from utils import plot_training_results, plot_comparison_results, evaluate_model
from optimizers import create_sgd_optimizer, create_parabolic_optimizer


def train_epoch_normal(model, trainloader, criterion, optimizer, device):
    """Train one epoch using standard SGD."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    train_loss = running_loss / len(trainloader.dataset)
    train_acc = 100. * correct / total
    return train_loss, train_acc


def train_epoch_parabolic(model, trainloader, criterion, parabolic_opt, device):
    """Train one epoch using parabolic approximation optimizer."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Use parabolic optimizer
        loss = parabolic_opt.step(criterion, inputs, labels)
        
        running_loss += loss * inputs.size(0)
        
        # Calculate accuracy
        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    train_loss = running_loss / len(trainloader.dataset)
    train_acc = 100. * correct / total
    return train_loss, train_acc


def train_normal(model, trainloader, testloader, criterion, optimizer, scheduler,
                 num_epochs, device):
    """Train model using standard SGD.
    
    Returns:
        train_losses, train_accs, test_accs
    """
    train_losses = []
    train_accs = []
    test_accs = []
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch_normal(
            model, trainloader, criterion, optimizer, device
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Evaluate on test set
        _, test_acc = evaluate_model(model, testloader, criterion, device)
        test_accs.append(test_acc)
        
        scheduler.step()
        print(f'Epoch [{epoch+1}/{num_epochs}] '
              f'Train Loss: {train_loss:.4f} | '
              f'Train Acc: {train_acc:.2f}% | '
              f'Test Acc: {test_acc:.2f}%')
    
    return train_losses, train_accs, test_accs


def train_parabolic(model, trainloader, testloader, criterion, parabolic_opt,
                    num_epochs, device):
    """Train model using parabolic approximation.
    
    Returns:
        train_losses, train_accs, test_accs
    """
    train_losses = []
    train_accs = []
    test_accs = []
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch_parabolic(
            model, trainloader, criterion, parabolic_opt, device
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Evaluate on test set
        _, test_acc = evaluate_model(model, testloader, criterion, device)
        test_accs.append(test_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] '
              f'Train Loss: {train_loss:.4f} | '
              f'Train Acc: {train_acc:.2f}% | '
              f'Test Acc: {test_acc:.2f}%')
    
    return train_losses, train_accs, test_accs


def main(args):
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    print("Loading CIFAR-10 dataset...")
    trainloader, testloader, classes = load_cifar10(
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        num_workers=args.num_workers
    )
    print(f"Classes: {classes}")
    
    # Store results for comparison
    results = {}
    
    # ========== Train with Normal SGD ==========
    if args.train_normal:
        print("\n" + "="*50)
        print("Training with Normal SGD...")
        print("="*50)
        
        model_normal = create_model(num_classes=10, device=device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer, scheduler = create_sgd_optimizer(
            model_normal, 
            lr=args.lr, 
            momentum=args.momentum, 
            weight_decay=args.weight_decay
        )
        
        normal_results = train_normal(
            model_normal, trainloader, testloader, criterion,
            optimizer, scheduler, args.epochs, device
        )
        results['Normal SGD'] = normal_results
        
        # Save model
        if args.save_model:
            torch.save(model_normal.state_dict(), 
                      f'resnet18_normal_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')
    
    # ========== Train with Parabolic Optimizer ==========
    if args.train_parabolic:
        print("\n" + "="*50)
        print("Training with Parabolic Approximation...")
        print("="*50)
        
        model_parabolic = create_model(num_classes=10, device=device)
        criterion = torch.nn.CrossEntropyLoss()
        parabolic_opt = create_parabolic_optimizer(
            model_parabolic, 
            epsilon=args.epsilon, 
            lr=args.parabolic_lr
        )
        
        parabolic_results = train_parabolic(
            model_parabolic, trainloader, testloader, criterion,
            parabolic_opt, args.epochs, device
        )
        results['Parabolic'] = parabolic_results
        
        # Save model
        if args.save_model:
            torch.save(model_parabolic.state_dict(), 
                      f'resnet18_parabolic_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')
    
    # ========== Plot Results ==========
    if results:
        print("\n" + "="*50)
        print("Plotting results...")
        print("="*50)
        
        # Plot individual results
        for method_name, (train_losses, train_accs, test_accs) in results.items():
            plot_training_results(
                train_losses, train_accs, test_accs,
                save_path=f'training_{method_name.replace(" ", "_").lower()}.png'
            )
        
        # Plot comparison
        if len(results) > 1:
            plot_comparison_results(results, metric='accuracy',
                                   save_path='comparison_accuracy.png')
            plot_comparison_results(results, metric='loss',
                                   save_path='comparison_loss.png')
    
    print("\nTraining complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ResNet18 Training with Normal and Parabolic Loss')
    
    # Training options
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Training batch size')
    parser.add_argument('--test-batch-size', type=int, default=100, help='Test batch size')
    parser.add_argument('--num-workers', type=int, default=2, help='Data loader workers')
    
    # Optimizer options
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate for SGD')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--parabolic-lr', type=float, default=0.01, help='Learning rate for parabolic')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon for parabolic approximation')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    # Training mode
    parser.add_argument('--train-normal', action='store_true', help='Train with normal SGD')
    parser.add_argument('--train-parabolic', action='store_true', help='Train with parabolic')
    
    # Save options
    parser.add_argument('--save-model', action='store_true', help='Save trained models')
    
    args = parser.parse_args()
    
    # Default: train both if no mode specified
    if not args.train_normal and not args.train_parabolic:
        args.train_normal = True
        args.train_parabolic = True
        print("No training mode specified, running both normal and parabolic training.")
    
    main(args)
