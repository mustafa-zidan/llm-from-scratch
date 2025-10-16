from typing import Any, SupportsIndex, List

import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, text: str, tokenizer: Any, max_length: int, stride: int):
        """
        Initialize the dataset with text, tokenizer, max_length, and stride.
        Note: tokenizer can be any object that has an encode method, so either our custom implementation or the tiktoken library.
        """
        self.input_ids: List[torch.Tensor] = []
        self.target_ids: List[torch.Tensor] = []

        token_ids = tokenizer.encode(text)

        for i in range(0, len(token_ids) - max_length, stride):
            # This uses a sliding window to chunk the book into overlapping sequences of max_length
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.input_ids)

    def __getitem__(self, idx: SupportsIndex) -> tuple[torch.Tensor, torch.Tensor]:
        """Return input and target tensors for a given index."""
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
        text: str,
        batch_size:int=4,
        max_length: int =265,
        stride:int=128,
        shuffle:bool=True,
        drop_last=True,
        num_workers:int=0
) -> DataLoader:
    """Create a PyTorch DataLoader for the given text."""
    dataset = GPTDatasetV1(
        text=text,
        tokenizer=tiktoken.encoding_for_model("gpt2"),
        max_length=max_length,
        stride=stride
    )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )