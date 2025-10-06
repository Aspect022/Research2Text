import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class SimpleTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 32,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask=None,
        tgt_mask=None,
        src_key_padding_mask=None,
        tgt_key_padding_mask=None,
    ) -> torch.Tensor:
        # src/tgt shape: (batch, seq_len)
        src_emb = self.embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        tgt_emb = self.embedding(tgt) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.pos_encoder(tgt_emb)
        # Transformer expects (seq_len, batch, d_model)
        src_emb = src_emb.transpose(0, 1)
        tgt_emb = tgt_emb.transpose(0, 1)
        output = self.transformer(
            src_emb,
            tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        # output shape: (seq_len, batch, d_model)
        output = self.fc_out(output)
        # return to (batch, seq_len, vocab_size)
        return output.transpose(0, 1)
