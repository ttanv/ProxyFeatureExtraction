"""
Simple Transformer-based classifier.
"""

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Injects some information about the relative or absolute position of the tokens in the sequence."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class SimpleTransformer(nn.Module):
    """A simple Transformer-based classifier."""
    def __init__(self, input_dim: int, num_heads: int, num_encoder_layers: int, num_classes: int, sequence_length: int):
        """
        Initializes the Transformer model.
        """
        super(SimpleTransformer, self).__init__()
        self.d_model = input_dim
        # This initial layer is used to project the input features to the model's dimension
        self.input_projection = nn.Linear(1, self.d_model) # Assuming raw signal comes in as (Batch, Seq, 1)
        self.pos_encoder = PositionalEncoding(self.d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        self.fc_out = nn.Linear(self.d_model, num_classes)
        self.sequence_length = sequence_length

    def forward(self, src):
        """
        Forward pass.
        `src` is expected to have shape (batch_size, sequence_length, features)
        """
        # Project input to the model dimension
        src = self.input_projection(src.permute(0, 2, 1)).permute(1, 0, 2) # (Seq, Batch, d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.mean(dim=0)  # Average pooling over the sequence
        output = self.fc_out(output)
        return output 