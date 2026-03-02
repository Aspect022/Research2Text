import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding.
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor of same shape with positional information added.
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int = 100, d_model: int = 32, nhead: int = 4,
                 num_encoder_layers: int = 2, num_decoder_layers: int = 2,
                 dim_feedforward: int = 64, dropout: float = 0.1):
        super().__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model
        self.src_embedding = nn.Embedding(vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout)
        self.generator = nn.Linear(d_model, vocab_size)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_key_padding_mask: torch.Tensor = None,
                tgt_key_padding_mask: torch.Tensor = None,
                memory_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass.
        Args:
            src: (batch, src_seq_len) LongTensor of token indices.
            tgt: (batch, tgt_seq_len) LongTensor of token indices.
        Returns:
            Logits of shape (batch, tgt_seq_len, vocab_size).
        """
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.pos_encoder(tgt_emb)

        # Transformer expects (seq_len, batch, d_model)
        src_emb = src_emb.transpose(0, 1)
        tgt_emb = tgt_emb.transpose(0, 1)
        output = self.transformer(src_emb, tgt_emb,
                                    src_key_padding_mask=src_key_padding_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask)
        output = output.transpose(0, 1)  # back to (batch, seq_len, d_model)
        return self.generator(output)
