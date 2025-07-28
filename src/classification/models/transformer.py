import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return x


class SimpleTransformerClassifier(nn.Module):
    """
    Expects input of shape (batch, seq_len, feat_dim)
    """

    def __init__(
        self,
        feat_dim: int,
        n_classes: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.fc_out = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        cls = self.cls_token.expand(B, -1, -1)          # (B,1,D)
        x = torch.cat([cls, x], 1)                       # prepend token
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        encoded = self.encoder(x)
        logits = self.fc_out(encoded[:, 0])              # use CLS token
        return logits
