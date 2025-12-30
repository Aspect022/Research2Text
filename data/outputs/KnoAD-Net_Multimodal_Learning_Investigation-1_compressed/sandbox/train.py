import torch
import torch.nn as nn
import torch.optim as optim
from model import TransformerModel
from utils import set_seed

def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
    return mask

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # hyperparameters
    src_vocab = 1000
    tgt_vocab = 1000
    seq_len = 20
    batch_size = 32
    epochs = 5
    lr = 1e-3

    model = TransformerModel(src_vocab, tgt_vocab).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        src = torch.randint(1, src_vocab, (seq_len, batch_size), device=device)
        tgt_input = torch.randint(1, tgt_vocab, (seq_len, batch_size), device=device)
        tgt_output = torch.randint(1, tgt_vocab, (seq_len, batch_size), device=device)

        src_mask = None
        tgt_mask = generate_square_subsequent_mask(seq_len).to(device)

        output = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)
        loss = criterion(output.view(-1, tgt_vocab), tgt_output.view(-1))
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

if __name__ == '__main__':
    main()
