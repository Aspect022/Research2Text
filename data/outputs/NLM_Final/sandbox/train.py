import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from model import SimpleTransformer
from utils import generate_synthetic_data

def main():
    # Hyperparameters
    vocab_size = 100
    num_classes = 10
    seq_len = 20
    batch_size = 32
    epochs = 5
    learning_rate = 1e-3

    # Model, loss, optimizer
    model = SimpleTransformer(vocab_size=vocab_size, max_seq_len=seq_len, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Synthetic dataset
    inputs, labels = generate_synthetic_data(num_samples=500, seq_len=seq_len, vocab_size=vocab_size, num_classes=num_classes)
    dataset = TensorDataset(inputs, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch_inputs, batch_labels in loader:
            optimizer.zero_grad()
            logits = model(batch_inputs)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_inputs.size(0)
        avg_loss = total_loss / len(dataset)
        print(f'Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}')

if __name__ == '__main__':
    main()