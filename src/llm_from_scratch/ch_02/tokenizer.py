"""
Simple tokenizer implementation for LLM from scratch - Chapter 1.
"""

import re
from typing import List, Dict, Tuple


class SimpleTokenizerV1:
    """A simple tokenizer that splits text into subword units."""
    
    def __init__(self, raw_text:str=''):
        self.vocab = {}
        self.inverse_vocab = {}
        self.vocab_size = 0
        self.build_vocab(raw_text)
        
    def build_vocab(self, text: str):
        """Build vocabulary from a list of texts."""
        # clean up the text
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        all_words = sorted(set(preprocessed))
        self.vocab = {token:index for index,token in enumerate(all_words)}
        self.inverse_vocab = {i:s for s,i in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        for i, item in enumerate(self.vocab.items()):
            print(item)
            if i >= 50:
                break

    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs."""
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        return [self.vocab[token] for token in preprocessed]
    
    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text."""
        text = " ".join([self.inverse_vocab[token_id] for token_id in token_ids ])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text



class SimpleTokenizerV2:
    """A simple tokenizer that splits text into subword units."""

    def __init__(self, raw_text:str=''):
        self.vocab = {}
        self.inverse_vocab = {}
        self.vocab_size = 0
        self.build_vocab(raw_text)

    def build_vocab(self, text: str):
        """Build vocabulary from a list of texts."""
        # clean up the text
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        all_words = sorted(set(preprocessed))
        all_words.extend(["<|endoftext|>", "<|unk|>"])
        self.vocab = {token:index for index,token in enumerate(all_words)}
        self.inverse_vocab = {i:s for s,i in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        for i, item in enumerate(self.vocab.items()):
            print(item)
            if i >= 50:
                break

    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs."""
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        return [self.vocab.get(token, self.vocab['<|unk|>']) for token in preprocessed]

    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text."""
        text = " ".join([self.inverse_vocab.get(token_id, '<|unk|>') for token_id in token_ids ])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text