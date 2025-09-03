# Cosine Learning Rate Schedule with Warmup

## Overview

Added support for cosine learning rate scheduling with warmup in the ParToken training pipeline. This feature provides more sophisticated learning rate scheduling that can improve model performance.

## Configuration

### New Parameters in `config_partoken.yaml`

```yaml
train:
  # ... other parameters ...
  # Cosine learning rate schedule with warmup
  use_cosine_schedule: false  # Set to true to enable cosine scheduling
  warmup_epochs: 5           # Number of warmup epochs
```

### Parameters Description

- **`use_cosine_schedule`**: Boolean flag to enable/disable cosine scheduling
  - `false` (default): Use constant learning rate
  - `true`: Use cosine annealing with warmup
  
- **`warmup_epochs`**: Number of epochs for learning rate warmup
  - During warmup: LR linearly increases from 1e-6 to the target LR
  - After warmup: LR follows cosine annealing schedule

## Learning Rate Schedule Formula

```python
def lr_lambda(epoch):
    if epoch < warmup_epochs:
        # Linear warmup
        return max(1e-06, epoch / max(1, warmup_epochs))
    
    # Cosine annealing
    progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * progress))
```

## Usage Examples

### 1. Default Training (Constant LR)
```bash
python run_partoken.py
# or explicitly
python run_partoken.py train.use_cosine_schedule=false
```

### 2. Enable Cosine Schedule with Default Warmup
```bash
python run_partoken.py train.use_cosine_schedule=true
```

### 3. Custom Warmup Duration
```bash
python run_partoken.py \
    train.use_cosine_schedule=true \
    train.warmup_epochs=10
```

### 4. Complete Training Configuration
```bash
python run_partoken.py \
    train.epochs=100 \
    train.lr=1e-3 \
    train.use_cosine_schedule=true \
    train.warmup_epochs=10
```

## Benefits

1. **Improved Convergence**: Warmup helps stabilize training in early epochs
2. **Better Final Performance**: Cosine annealing can lead to better final model performance
3. **Reduced Overfitting**: Gradually decreasing learning rate helps prevent overfitting
4. **Stable Training**: Smooth learning rate transitions throughout training

## Schedule Visualization

The learning rate follows this pattern:
- **Epochs 0 → warmup_epochs**: Linear increase from ~0 to target LR
- **Epochs warmup_epochs → max_epochs**: Cosine decay from target LR to ~0

## Integration with PyTorch Lightning

The scheduler is integrated using PyTorch Lightning's scheduler configuration:

```python
{
    'optimizer': optimizer,
    'lr_scheduler': {
        'scheduler': scheduler,
        'interval': 'epoch',      # Update per epoch
        'frequency': 1            # Update every epoch
    }
}
```

## Testing

Use the provided test script to verify functionality:
```bash
bash test_cosine_scheduler.sh
```
