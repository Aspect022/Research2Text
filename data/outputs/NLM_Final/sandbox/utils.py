import torch

def generate_synthetic_data(num_samples=1000, seq_len=20, vocab_size=100, num_classes=10):
    """
    Generate random integer sequences and classification labels.

    Returns:
        inputs: LongTensor of shape (num_samples, seq_len)
        labels: LongTensor of shape (num_samples,)
    """
    inputs = torch.randint(0, vocab_size, (num_samples, seq_len), dtype=torch.long)
    labels = torch.randint(0, num_classes, (num_samples,), dtype=torch.long)
    return inputs, labels