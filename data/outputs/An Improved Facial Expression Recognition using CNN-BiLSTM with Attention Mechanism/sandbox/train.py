import torch
import torch.nn as nn
import torch.optim as optim
from model import SimpleTransformer
from utils import generate_random_batch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper‑parameters
VOCAB_SIZE = 100  # include PAD token (0)
SEQ_LEN = 12
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-3
PAD_IDX = 0

model = SimpleTransformer(vocab_size=VOCAB_SIZE, d_model=32).to(DEVICE)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(1, EPOCHS + 1):
    model.train()
    src, tgt_input, tgt_output = generate_random_batch(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE, PAD_IDX)
    src = src.to(DEVICE)
    tgt_input = tgt_input.to(DEVICE)
    tgt_output = tgt_output.to(DEVICE)

    # Create padding masks (True where padding)
    src_key_padding_mask = (src == PAD_IDX)
    tgt_key_padding_mask = (tgt_input == PAD_IDX)
    memory_key_padding_mask = src_key_padding_mask

    optimizer.zero_grad()
    logits = model(src, tgt_input,
                   src_key_padding_mask=src_key_padding_mask,
                   tgt_key_padding_mask=tgt_key_padding_mask,
                   memory_key_padding_mask=memory_key_padding_mask)
    # logits shape: (batch, seq_len, vocab_size) -> reshape for loss
    loss = criterion(logits.view(-1, VOCAB_SIZE), tgt_output.view(-1))
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}/{EPOCHS} - Loss: {loss.item():.4f}")

print('Training complete.')
