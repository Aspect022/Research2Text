import torch

def generate_random_batch(seq_len: int, batch_size: int, vocab_size: int):
    """Generate a random batch of source and target sequences.
    Returns:
        src: LongTensor of shape (batch_size, seq_len)
        tgt_input: LongTensor of shape (batch_size, seq_len) (input to decoder)
        tgt_output: LongTensor of shape (batch_size, seq_len) (labels for loss)
    """
    src = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    # For teacher forcing we shift the target by one token
    tgt = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    tgt_input = tgt[:, :-1]
    tgt_output = tgt[:, 1:]
    return src, tgt_input, tgt_output
