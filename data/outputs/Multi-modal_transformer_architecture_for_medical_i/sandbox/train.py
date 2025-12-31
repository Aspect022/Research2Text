import torch
import torch.nn as nn
import torch.optim as optim
from model import SimpleTransformer
from utils import get_dataloader

def train():
    # Hyperparameters (fallback defaults)
    vocab_size = 100
    num_classes = 2
    seq_len = 20
    batch_size = 32
    epochs = 5
    learning_rate = 1e-3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleTransformer(vocab_size=vocab_size, num_classes=num_classes, max_len=seq_len).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loader = get_dataloader(batch_size=batch_size)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch_inputs, batch_labels in loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_inputs.size(0)

        avg_loss = total_loss / len(loader.dataset)
        print(f'Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}')

if __name__ == '__main__':
    train()
