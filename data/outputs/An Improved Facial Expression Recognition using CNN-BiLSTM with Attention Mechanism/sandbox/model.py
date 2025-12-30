import torch
import torch.nn as nn
import math

class SimpleTransformer(nn.Module):
    """A minimal Transformer model for sequence‑to‑sequence tasks.

    It consists of an embedding layer, a positional encoding, a
    ``nn.Transformer`` module and a final linear layer projecting to the
    vocabulary size.
    """
    def __init__(self, vocab_size: int, d_model: int = 32, nhead: int = 4,
                 num_encoder_layers: int = 2, num_decoder_layers: int = 2,
                 dim_feedforward: int = 64, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout,
                                          batch_first=True)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_key_padding_mask: torch.Tensor = None,
                tgt_key_padding_mask: torch.Tensor = None,
                memory_key_padding_mask: torch.Tensor = None):
        # src, tgt: (batch, seq_len)
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model))
        output = self.transformer(src_emb, tgt_emb,
                                  src_key_padding_mask=src_key_padding_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask)
        return self.fc_out(output)  # (batch, tgt_seq_len, vocab_size)

class PositionalEncoding(nn.Module):
    """Injects information about the relative or absolute position of the tokens.

    Implementation follows the original Transformer paper.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
