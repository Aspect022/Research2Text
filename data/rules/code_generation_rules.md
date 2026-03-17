# Code Generation Rules

## Version
- Rules Version: 1.0.0
- Last Updated: 2026-03-17
- Compatible Models: qwen3-coder:480b-cloud, deepseek-v3.1:671b-cloud

## Table of Contents
1. [Core Principles](#core-principles)
2. [PyTorch Patterns](#pytorch-patterns)
3. [Architecture Templates](#architecture-templates)
4. [Training Loop Patterns](#training-loop-patterns)
5. [Dataset Loading Patterns](#dataset-loading-patterns)
6. [Common Mistakes](#common-mistakes)
7. [Best Practices](#best-practices)
8. [Examples](#examples)

---

## Core Principles

### Principle 1: Paper Fidelity
Generate code that exactly implements the paper's method. Do not add features not mentioned. Do not use modern alternatives unless paper specifies.

### Principle 2: Working Code
Generated code must be runnable. Include all imports. Handle edge cases. Add device management (CPU/GPU).

### Principle 3: Clarity Over Cleverness
Write clear, readable code. Use descriptive variable names. Add comments for complex operations. Follow PEP 8 style.

### Principle 4: Reproducibility
Set random seeds. Use deterministic operations where possible. Document hyperparameters. Save configuration.

---

## PyTorch Patterns

### Pattern 1: Model Definition

**Standard Structure:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelName(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Layer definitions

    def forward(self, x):
        # Forward pass
        return output
```

**Rules:**
- Always inherit from `nn.Module`
- Initialize parent class
- Store config as attribute
- Define layers in `__init__`, use in `forward`

### Pattern 2: Convolutional Layers

**Standard Conv Block:**
```python
nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
    nn.BatchNorm2d(out_channels),
    nn.ReLU(inplace=True)
)
```

**Rules:**
- Use `BatchNorm` after Conv (unless paper says otherwise)
- Use `inplace=True` for activation to save memory
- Calculate padding for "same" output: `padding = kernel_size // 2`

### Pattern 3: Residual Connections

**Standard Residual Block:**
```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out
```

**Rules:**
- Save residual before transformation
- Add back after transformation
- Apply activation after addition

### Pattern 4: Attention Mechanisms

**Multi-Head Attention:**
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, V)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.W_o(attn_output)
```

**Rules:**
- Ensure d_model divisible by num_heads
- Scale by sqrt(d_k)
- Use transpose for multi-head splitting
- Contiguous before view

---

## Architecture Templates

### Template 1: CNN Classifier

```python
class CNNClassifier(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

### Template 2: Transformer Encoder

```python
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8,
                 num_layers=6, d_ff=2048, dropout=0.1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
```

### Template 3: LSTM Sequence Model

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256,
                 num_layers=2, num_classes=10, bidirectional=True,
                 dropout=0.5):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )

        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(x)

        # Use final hidden state
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]

        return self.classifier(hidden)
```

---

## Training Loop Patterns

### Pattern 1: Standard Training Loop

```python
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        # Forward
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        # Backward
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy
```

### Pattern 2: Validation Loop

```python
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy
```

### Pattern 3: Learning Rate Scheduling

```python
# Cosine annealing
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs, eta_min=1e-6
)

# Step decay
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=30, gamma=0.1
)

# Reduce on plateau
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=10
)

# Usage in training
for epoch in range(num_epochs):
    train(...)
    val_loss = validate(...)
    scheduler.step(val_loss)  # or scheduler.step() for others
```

### Pattern 4: Early Stopping

```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif (self.mode == 'min' and score > self.best_score - self.min_delta) or \
             (self.mode == 'max' and score < self.best_score + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
```

---

## Dataset Loading Patterns

### Pattern 1: Image Dataset

```python
from torchvision import datasets, transforms

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.CIFAR10(
    root='./data', train=True, download=True, transform=train_transform
)
val_dataset = datasets.CIFAR10(
    root='./data', train=False, download=True, transform=val_transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
)
```

### Pattern 2: Custom Dataset

```python
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        # Load file paths and labels
        samples = []
        with open(self.data_path, 'r') as f:
            for line in f:
                path, label = line.strip().split(',')
                samples.append((path, int(label)))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
```

---

## Common Mistakes

### Mistake 1: Forgetting to Call optimizer.zero_grad()
**Wrong:**
```python
loss.backward()
optimizer.step()
```
**Right:**
```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### Mistake 2: Not Setting model.train() / model.eval()
**Wrong:**
```python
# Training
output = model(data)  # Missing model.train()

# Validation
output = model(data)  # Missing model.eval()
```
**Right:**
```python
# Training
model.train()
output = model(data)

# Validation
model.eval()
with torch.no_grad():
    output = model(data)
```

### Mistake 3: Wrong Loss Function for Problem Type
**Classification:** Use `CrossEntropyLoss` (not `MSELoss`)
**Binary Classification:** Use `BCEWithLogitsLoss` (not `BCELoss` with sigmoid)
**Regression:** Use `MSELoss` or `L1Loss`

### Mistake 4: Not Handling Device
**Wrong:**
```python
model = Model()
data = torch.randn(32, 3, 224, 224)
output = model(data)  # May fail if model on GPU, data on CPU
```
**Right:**
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model().to(device)
data = data.to(device)
output = model(data)
```

### Mistake 5: Using Softmax with CrossEntropyLoss
**Wrong:**
```python
output = F.softmax(model(data), dim=1)
loss = F.cross_entropy(output, target)
```
**Right:**
```python
output = model(data)
loss = F.cross_entropy(output, target)  # CrossEntropy includes softmax
```

---

## Best Practices

### Practice 1: Reproducibility
```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### Practice 2: Gradient Clipping
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Practice 3: Mixed Precision Training
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in dataloader:
    optimizer.zero_grad()

    with autocast():
        output = model(data)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Practice 4: Checkpointing
```python
def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']
```

---

## Examples

### Example 1: Complete Training Script

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Set seed for reproducibility
torch.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters (from paper)
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
NUM_CLASSES = 10

# Data loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model (from paper specification)
class PaperModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Define architecture based on paper
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(128 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

model = PaperModel(NUM_CLASSES).to(device)

# Loss and optimizer (from paper)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

# Save model
torch.save(model.state_dict(), 'model.pth')
```

---

*End of Code Generation Rules*
