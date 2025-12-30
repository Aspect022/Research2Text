import torch

def generate_random_batch(batch_size: int, seq_len: int, vocab_size: int,
                          pad_idx: int = 0) -> tuple:
    """Creates a random (src, tgt_input, tgt_output) batch.

    * ``src`` – source sequence.
    * ``tgt_input`` – target sequence shifted right (for teacher forcing).
    * ``tgt_output`` – target sequence (the ground‑truth we predict).
    """
    src = torch.randint(1, vocab_size, (batch_size, seq_len), dtype=torch.long)
    tgt = torch.randint(1, vocab_size, (batch_size, seq_len), dtype=torch.long)
    # Use 0 as padding token – here we randomly replace some positions with pad_idx
    mask = torch.rand_like(src, dtype=torch.float) < 0.1  # 10% padding
    src[mask] = pad_idx
    tgt[mask] = pad_idx
    # Shift target for decoder input
    tgt_input = torch.clone(tgt)
    tgt_input[:, 1:] = tgt[:, :-1]
    tgt_input[:, 0] = pad_idx  # start token is padding token for simplicity
    return src, tgt_input, tgt
