"""
Simple tokenizer implementation for LLM from scratch - Chapter 1.
"""

import re
from typing import List, Dict, Tuple


class SimpleTokenizer:
    """A simple tokenizer that splits text into subword units."""
    
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}
        self.vocab_size = 0
        
    def build_vocab(self, texts: List[str]):
        """Build vocabulary from a list of texts."""
        # Simple character-level tokenization
        all_chars = set()
        for text in texts:
            all_chars.update(text)
        
        # Add special tokens
        special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        self.vocab = {token: i for i, token in enumerate(special_tokens)}
        self.vocab_size = len(special_tokens)
        
        # Add character tokens
        for char in sorted(all_chars):
            self.vocab[char] = self.vocab_size
            self.vocab_size += 1
            
        # Create inverse vocabulary
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs."""
        return [self.vocab.get(char, self.vocab['<UNK>']) for char in text]
    
    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text."""
        return ''.join([self.inverse_vocab.get(token_id, '<UNK>') 
                       for token_id in token_ids])
        
    def get_vocab_size(self) -> int:
        """Return the vocabulary size."""
        return self.vocab_size