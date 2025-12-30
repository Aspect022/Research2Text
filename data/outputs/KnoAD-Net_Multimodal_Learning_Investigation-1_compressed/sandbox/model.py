import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create constant "pe" matrix with values dependent on
        # position and i (dimension).
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Shape (max_len, d_model) -> we will expand a batch dimension later.
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Add positional encoding.

        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        Returns:
            Tensor of same shape as ``x`` with positional information added.
        """
        # x shape: (seq_len, batch, d_model)
        seq_len = x.size(0)
        # self.pe shape: (1, max_len, d_model) -> slice to (1, seq_len, d_model)
        pe_slice = self.pe[:, :seq_len, :]  # (1, seq_len, d_model)
        # Rearrange to (seq_len, 1, d_model) so it broadcasts over the batch dimension.
        pe_slice = pe_slice.permute(1, 0, 2)  # (seq_len, 1, d_model)
        x = x + pe_slice
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size=1000, tgt_vocab_size=1000, d_model=512, nhead=8,
                 num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout)
        self.generator = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        # src & tgt shape: (seq_len, batch)
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.transformer.d_model))
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.transformer.d_model))
        output = self.transformer(src_emb, tgt_emb,
                                  src_mask=src_mask,
                                  tgt_mask=tgt_mask,
                                  src_key_padding_mask=src_key_padding_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask)
        return self.generator(output)
