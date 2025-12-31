import torch
from torch.utils.data import TensorDataset, DataLoader

def generate_synthetic_data(num_samples=1000, seq_len=20, vocab_size=100, num_classes=2):
    """Generate random integer sequences and labels for placeholder training."""
    inputs = torch.randint(0, vocab_size, (num_samples, seq_len), dtype=torch.long)
    labels = torch.randint(0, num_classes, (num_samples,), dtype=torch.long)
    return inputs, labels

def get_dataloader(batch_size=32):
    inputs, labels = generate_synthetic_data()
    dataset = TensorDataset(inputs, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader
