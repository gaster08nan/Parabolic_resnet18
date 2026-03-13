# ResNet18 Training with Normal and Parabolic Optimization

This project implements ResNet18 for CIFAR-10 image classification with two optimization methods:
1. **Normal SGD** - Standard Stochastic Gradient Descent with momentum
2. **Parabolic Approximation** - Custom optimizer using parabolic loss approximation

## Project Structure

```
homework_resnet18/
├── model.py          # ResNet18 architecture
├── data.py           # CIFAR-10 data loading and transforms
├── optimizers.py     # SGD and ParabolicOptimizer implementations
├── utils.py          # Plotting and evaluation utilities
├── train.py          # Main training script
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, for faster training)

## Installation

### Step 1: Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start - Train Both Methods

Run training with both Normal SGD and Parabolic optimization:

```bash
python train.py
```

This will:
- Train ResNet18 using Normal SGD (10 epochs)
- Train ResNet18 using Parabolic Approximation (10 epochs)
- Generate comparison plots for accuracy and loss

### Train with Specific Method

**Only Normal SGD:**
```bash
python train.py --train-normal
```

**Only Parabolic Optimization:**
```bash
python train.py --train-parabolic
```

### Custom Hyperparameters

```bash
python train.py \
    --epochs 20 \
    --batch-size 64 \
    --lr 0.05 \
    --parabolic-lr 0.01 \
    --epsilon 0.05 \
    --save-model
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 10 | Number of training epochs |
| `--batch-size` | 128 | Training batch size |
| `--test-batch-size` | 100 | Test/validation batch size |
| `--num-workers` | 2 | Number of data loader workers |
| `--lr` | 0.1 | Learning rate for SGD |
| `--momentum` | 0.9 | Momentum for SGD |
| `--weight-decay` | 5e-4 | Weight decay (L2 regularization) |
| `--parabolic-lr` | 0.01 | Learning rate for parabolic optimizer |
| `--epsilon` | 0.1 | Perturbation for parabolic approximation |
| `--device` | cuda | Device to use (cuda/cpu) |
| `--train-normal` | - | Train with Normal SGD only |
| `--train-parabolic` | - | Train with Parabolic only |
| `--save-model` | - | Save trained model checkpoints |

## Output

After training, the script generates:

1. **Console Output** - Epoch-by-epoch training loss and accuracy
2. **Plots**:
   - `training_normal_sgd.png` - Loss and accuracy curves for Normal SGD
   - `training_parabolic.png` - Loss and accuracy curves for Parabolic
   - `comparison_accuracy.png` - Accuracy comparison between methods
   - `comparison_loss.png` - Loss comparison between methods
3. **Model Checkpoints** (if `--save-model`):
   - `resnet18_normal_YYYYMMDD_HHMMSS.pth`
   - `resnet18_parabolic_YYYYMMDD_HHMMSS.pth`

## Example Output

```
Using device: cuda
Loading CIFAR-10 dataset...
Classes: ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

==================================================
Training with Normal SGD...
==================================================
Epoch [1/10] Train Loss: 1.5234 | Train Acc: 45.23% | Test Acc: 52.10%
Epoch [2/10] Train Loss: 0.9876 | Train Acc: 65.45% | Test Acc: 68.32%
...

==================================================
Training with Parabolic Approximation...
==================================================
Epoch [1/10] Train Loss: 1.6543 | Train Acc: 42.10% | Test Acc: 48.50%
...
```

## How Parabolic Optimization Works

The parabolic optimizer approximates the loss function as a parabola around current weights:

1. **Evaluate loss** at current weights: L(w)
2. **Perturb weights** by +ε and -ε to get L(w+ε) and L(w-ε)
3. **Fit parabola**: L(w) ≈ aw² + bw + c
   - `a = (L+ + L- - 2L₀) / (2ε²)`
   - `b = (L+ - L-) / (2ε)`
4. **Find minimum**: w* = -b / (2a)
5. **Update weights**: w_new = w + shift

## Notes

- First run will download CIFAR-10 dataset (~170MB) to `./data/`
- Training time: ~5-10 minutes per method on GPU, ~30-60 minutes on CPU
- For CPU-only training: `python train.py --device cpu`
- Reduce batch size if running out of GPU memory: `--batch-size 32`

## License

This project is for educational purposes.
