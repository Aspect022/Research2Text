import torch
import torch.nn as nn
import torch.optim as optim
from model import SimpleTransformer
from utils import generate_random_batch

# Hyperparameters
VOCAB_SIZE = 100
SEQ_LEN = 16
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-3

# Model, loss, optimizer
model = SimpleTransformer(vocab_size=VOCAB_SIZE)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # assuming 0 is pad token
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

model.train()
for epoch in range(1, EPOCHS + 1):
    src, tgt_input, tgt_output = generate_random_batch(SEQ_LEN, BATCH_SIZE, VOCAB_SIZE)
    optimizer.zero_grad()
    # Forward pass
    logits = model(src, tgt_input)  # logits shape: (batch, seq_len-1, vocab_size)
    # Reshape for loss: (batch*seq_len-1, vocab_size)
    loss = criterion(logits.view(-1, VOCAB_SIZE), tgt_output.contiguous().view(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}/{EPOCHS} - Loss: {loss.item():.4f}")
