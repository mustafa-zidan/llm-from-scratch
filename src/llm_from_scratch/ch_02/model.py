"""
Basic LLM model implementation from scratch - Chapter 2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SimpleLLM(nn.Module):
    """A simple LLM implementation with attention mechanism."""
    
    def __init__(self, vocab_size: int, embed_dim: int = 128, num_heads: int = 4, 
                 num_layers: int = 2, max_seq_len: int = 512):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # Embedding layer
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.randn(max_seq_len, embed_dim))
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(embed_dim, num_heads) 
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """Forward pass through the model."""
        # Embedding
        x = self.token_embedding(x) + self.positional_encoding[:x.size(1)]
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Output
        logits = self.lm_head(x)
        
        if targets is not None:
            # Calculate loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), 
                                 targets.view(-1), ignore_index=-1)
            return logits, loss
        
        return logits


class TransformerLayer(nn.Module):
    """A single transformer layer."""
    
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor):
        """Forward pass through the transformer layer."""
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.ln1(x + attn_out)
        
        # Feed-forward
        ff_out = self.feed_forward(x)
        x = self.ln2(x + ff_out)
        
        return x