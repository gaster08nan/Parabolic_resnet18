import matplotlib.pyplot as plt
import torch


def plot_training_results(train_losses, train_acc_list, test_acc_list, save_path=None):
    """Plot training loss and accuracy curves.
    
    Args:
        train_losses: List of training losses per epoch
        train_acc_list: List of training accuracies per epoch
        test_acc_list: List of test accuracies per epoch
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(12, 5))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_list, label='Train Accuracy')
    plt.plot(test_acc_list, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_comparison_results(results_dict, metric='accuracy', save_path=None):
    """Plot comparison of different training methods.
    
    Args:
        results_dict: Dictionary with method names as keys and 
                      (train_losses, train_accs, test_accs) as values
        metric: 'accuracy' or 'loss'
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(10, 6))
    
    for method_name, (train_losses, train_accs, test_accs) in results_dict.items():
        if metric == 'accuracy':
            plt.plot(test_accs, label=f'{method_name} Test Acc')
            plt.plot(train_accs, label=f'{method_name} Train Acc', linestyle='--')
        else:
            plt.plot(train_losses, label=f'{method_name} Loss')
    
    plt.xlabel('Epochs')
    plt.ylabel(metric.capitalize())
    plt.title(f'Training Comparison - {metric.capitalize()}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    
    plt.show()


def evaluate_model(model, dataloader, criterion, device):
    """Evaluate model on a dataset.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for the dataset
        criterion: Loss function
        device: Device to run evaluation on
        
    Returns:
        average_loss, accuracy
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy
