import torch
import torch.nn as nn
import torch.optim as optim


class ParabolicOptimizer:
    """Optimizer using parabolic approximation for weight updates.
    
    This optimizer approximates the loss function as a parabola around
    the current weights and finds the minimum of the parabola to update.
    """
    
    def __init__(self, model, epsilon=0.1, lr=0.01):
        """
        Args:
            model: The model to optimize
            epsilon: Small perturbation for numerical differentiation
            lr: Learning rate for the update
        """
        self.model = model
        self.epsilon = epsilon
        self.lr = lr
        self.params = list(model.parameters())
        
    def step(self, loss_fn, inputs, labels):
        """Perform one optimization step using parabolic approximation.
        
        Args:
            loss_fn: Loss function to minimize
            inputs: Input batch
            labels: Target labels
            
        Returns:
            The loss value after update
        """
        self.model.zero_grad()
        
        # Get current loss
        outputs = self.model(inputs)
        base_loss = loss_fn(outputs, labels)
        
        # For each parameter, estimate parabolic approximation
        with torch.no_grad():
            for param in self.params:
                if param.grad is None:
                    continue
                    
                # Compute gradient at current point
                grad = param.grad
                
                # Compute loss at perturbed points for curvature estimation
                # L(w + epsilon)
                param.add_(self.epsilon)
                outputs_plus = self.model(inputs)
                loss_plus = loss_fn(outputs_plus, labels)
                
                # L(w - epsilon)
                param.add_(-2 * self.epsilon)
                outputs_minus = self.model(inputs)
                loss_minus = loss_fn(outputs_minus, labels)
                
                # Restore original
                param.add_(self.epsilon)
                
                # Parabolic approximation: L(w) ≈ a*w^2 + b*w + c
                # a = (L+ + L- - 2*L0) / (2*epsilon^2)
                # b = (L+ - L-) / (2*epsilon)
                epsilon_sq = self.epsilon ** 2
                a = (loss_plus.item() + loss_minus.item() - 2 * base_loss.item()) / (2 * epsilon_sq)
                b = (loss_plus.item() - loss_minus.item()) / (2 * self.epsilon)
                
                # Optimal shift: w* = -b / (2a)
                if abs(a) > 1e-8:
                    shift = -b / (2 * a)
                    # Clip shift to prevent large updates
                    shift = torch.clamp(shift, -1.0, 1.0)
                    param.add_(shift * self.lr)
                else:
                    # Fall back to gradient descent if parabola is too flat
                    param.add_(-grad * self.lr)
        
        return base_loss.item()


def create_sgd_optimizer(model, lr=0.1, momentum=0.9, weight_decay=5e-4):
    """Create standard SGD optimizer with momentum.
    
    Args:
        model: Model to optimize
        lr: Learning rate
        momentum: Momentum factor
        weight_decay: L2 regularization
        
    Returns:
        optimizer, scheduler
    """
    optimizer = optim.SGD(
        model.parameters(), 
        lr=lr, 
        momentum=momentum, 
        weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    return optimizer, scheduler


def create_parabolic_optimizer(model, epsilon=0.1, lr=0.01):
    """Create parabolic approximation optimizer.
    
    Args:
        model: Model to optimize
        epsilon: Perturbation for numerical differentiation
        lr: Learning rate
        
    Returns:
        ParabolicOptimizer instance
    """
    return ParabolicOptimizer(model, epsilon=epsilon, lr=lr)
