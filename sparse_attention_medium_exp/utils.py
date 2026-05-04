import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer


class WikiTextDataset(Dataset):
    def __init__(self, split: str, tokenizer, seq_len: int = 256):
        # 使用 WikiText-103（约 1.03 亿 token），远大于 WikiText-2（约 200 万 token）
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
        tokenized = dataset.map(
            lambda x: tokenizer(x["text"], truncation=True, max_length=None, return_overflowing_tokens=True),
            batched=True,
            remove_columns=dataset.column_names
        )
        # 展平所有 token
        all_ids = []
        for sample in tokenized:
            all_ids.extend(sample["input_ids"])
        self.data = torch.tensor(all_ids, dtype=torch.long)
        self.seq_len = seq_len
        print(f"WikiText-103 {split}: {len(self.data)} tokens loaded")

    def __len__(self):
        return max(0, len(self.data) - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1  # 额外一个 token 作为目标
        chunk = self.data[start:end]
        input_ids = chunk[:-1]
        targets = chunk[1:]
        return input_ids, targets


def get_dataloaders(tokenizer, seq_len: int = 256, batch_size: int = 4):
    train_dataset = WikiTextDataset("train", tokenizer, seq_len)
    val_dataset = WikiTextDataset("validation", tokenizer, seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
