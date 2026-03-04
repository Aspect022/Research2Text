# Code Generation Reference Rules
> This file is automatically injected into every code generation prompt.
> It ensures consistent, high-quality PyTorch output.

## 1. Project Structure Rules

Every generated project MUST contain:
- `model.py` — Model class(es) with proper `__init__` and `forward`
- `train.py` — Complete training loop with validation
- `utils.py`  — Helper functions (data loading, metrics, visualization)
- `requirements.txt` — All required packages

## 2. PyTorch Model Patterns

### Model Class Template
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelName(nn.Module):
    """One-line description of the architecture."""
    
    def __init__(self, num_classes: int, input_channels: int = 3, **kwargs):
        super().__init__()
        # Define all layers here, in sequential order
        # Use nn.ModuleList / nn.Sequential for repeated blocks
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Clear, step-by-step forward pass
        # Add shape comments: # (B, C, H, W) -> (B, C', H', W')
        return x
```

### Common Layer Recipes

**CNN Feature Extractor:**
```python
self.features = nn.Sequential(
    nn.Conv2d(in_ch, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),
    nn.Conv2d(64, 128, kernel_size=3, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),
)
```

**BiLSTM Layer:**
```python
self.bilstm = nn.LSTM(
    input_size=embed_dim,
    hidden_size=hidden_dim,
    num_layers=num_layers,
    batch_first=True,
    bidirectional=True,
    dropout=dropout if num_layers > 1 else 0.0,
)
# Output shape: (batch, seq_len, hidden_dim * 2)
```

**Multi-Head Self-Attention:**
```python
self.attention = nn.MultiheadAttention(
    embed_dim=d_model,
    num_heads=num_heads,
    dropout=dropout,
    batch_first=True,
)
# Usage: attn_out, attn_weights = self.attention(query, key, value)
```

**Transformer Encoder Block:**
```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=d_model,
    nhead=num_heads,
    dim_feedforward=d_ff,
    dropout=dropout,
    batch_first=True,
)
self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
```

**Vision Transformer Patch Embedding:**
```python
self.patch_embed = nn.Conv2d(
    in_channels, embed_dim,
    kernel_size=patch_size, stride=patch_size
)
# (B, 3, 224, 224) -> (B, embed_dim, 14, 14) -> flatten -> (B, 196, embed_dim)
self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
```

**Residual Block:**
```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.block(x) + x)
```

**Squeeze-and-Excitation Block:**
```python
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w
```

## 3. Training Loop Template

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# ─── Reproducibility ───
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed()

# ─── Device ───
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Model, Loss, Optimizer ───
model = ModelName(num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ─── Training Loop ───
best_val_acc = 0.0
for epoch in range(1, EPOCHS + 1):
    # --- Train ---
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
        train_correct += outputs.argmax(dim=1).eq(targets).sum().item()
        train_total += inputs.size(0)
    
    scheduler.step()
    
    # --- Validate ---
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
            val_correct += outputs.argmax(dim=1).eq(targets).sum().item()
            val_total += inputs.size(0)
    
    train_acc = 100.0 * train_correct / train_total
    val_acc = 100.0 * val_correct / val_total
    print(f"Epoch {epoch}/{EPOCHS} | "
          f"Train Loss: {train_loss/train_total:.4f} Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss/val_total:.4f} Acc: {val_acc:.2f}%")
    
    # --- Checkpoint ---
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print(f"  → Saved best model (val_acc={val_acc:.2f}%)")

print(f"Training complete. Best validation accuracy: {best_val_acc:.2f}%")
```

## 4. Dataset Loading Patterns

**Image Classification (torchvision):**
```python
from torchvision import datasets, transforms

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

**Custom Dataset:**
```python
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transform=None):
        self.samples = self._load_data(data_path)
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        if self.transform:
            x = self.transform(x)
        return x, y
```

## 5. Code Quality Rules

1. **ALWAYS** implement the EXACT architecture described in the paper — never substitute with a generic model
2. **ALWAYS** set random seeds for reproducibility
3. **ALWAYS** use `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
4. **ALWAYS** include type hints on function signatures
5. **ALWAYS** add shape comments in forward pass: `# (B, C, H, W)`
6. **ALWAYS** include docstrings for model classes explaining the architecture
7. **ALWAYS** use gradient clipping in training loops
8. **ALWAYS** implement both training and validation in the training loop
9. **ALWAYS** save the best model checkpoint
10. **NEVER** use random/dummy data — use real datasets (torchvision, torchaudio, etc.) or synthetic data that matches the paper's description
11. **NEVER** hardcode absolute file paths
12. **NEVER** leave placeholder comments like "# TODO" or "# implement this"
13. **NEVER** use deprecated PyTorch APIs

## 6. Architecture-Specific Guidelines

### CNN Architectures
- Use BatchNorm after every Conv layer (before activation)
- Use proper padding to maintain spatial dimensions when needed
- Use AdaptiveAvgPool2d before the classifier head

### LSTM / BiLSTM
- Always use `batch_first=True`
- For classification, use the last hidden state or attention-pooled output
- If bidirectional, remember output dim is `hidden_dim * 2`

### Transformer / ViT
- Use proper positional encoding (sinusoidal or learned)
- Implement proper attention masking if needed
- Use `batch_first=True` for consistency

### Attention Mechanisms
- Implement proper scaled dot-product attention: `softmax(QK^T / sqrt(d_k)) * V`
- For custom attention, always include the scaling factor
- Return attention weights when useful for visualization

### GAN Architectures
- Separate Generator and Discriminator classes
- Use proper weight initialization (Xavier or He)
- Implement proper training alternation
