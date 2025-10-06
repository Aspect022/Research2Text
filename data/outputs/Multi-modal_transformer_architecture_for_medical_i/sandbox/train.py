import torch
import torch.nn as nn
import torch.optim as optim
from model import SimpleTransformer


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Creates an upper‑triangular mask to hide future tokens.

    The mask is filled with ``-inf`` above the main diagonal so that the
    transformer does not attend to future positions.
    """
    mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
    return mask


def main():
    vocab_size = 1000
    seq_len = 20
    batch_size = 32
    epochs = 5

    model = SimpleTransformer(vocab_size=vocab_size)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, epochs + 1):
        # Random synthetic data (tokens in 1..vocab_size-1, 0 used as padding token)
        src = torch.randint(1, vocab_size, (batch_size, seq_len))
        tgt_input = torch.randint(1, vocab_size, (batch_size, seq_len))
        tgt_output = torch.randint(1, vocab_size, (batch_size, seq_len))

        tgt_mask = generate_square_subsequent_mask(seq_len)

        optimizer.zero_grad()
        output = model(src, tgt_input, tgt_mask=tgt_mask)
        # ``output`` has shape (batch, seq_len, vocab_size). ``view`` cannot be used
        # on a non‑contiguous tensor (the result of ``transpose`` in the model).
        # ``reshape`` works regardless of contiguity, or we could call
        # ``output.contiguous().view``. ``reshape`` is the most concise.
        loss = criterion(output.reshape(-1, vocab_size), tgt_output.reshape(-1))
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
