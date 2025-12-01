import torch
from torch.utils.data import Dataset, DataLoader
import os
from tokenizer import CustomBPE

class CPPCodeDataset(Dataset):
    def __init__(self, txt_file, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        if not os.path.exists(txt_file):
            print("Error, There is no data!")
            text = ""
        else:
            with open(txt_file, "r", encoding="utf-8") as f:
                text = f.read()

        print(f" '{txt_file}' tokenizing...")

        self.tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        print(f"Data is loaded! Tokens: {len(self.tokens)}")

    def __len__(self):
        if len(self.tokens) <= self.max_seq_len:
            return 0
        return len(self.tokens) - self.max_seq_len - 1

    def __getitem__(self, idx):
        x = self.tokens[idx : idx + self.max_seq_len]
        y = self.tokens[idx + 1 : idx + self.max_seq_len + 1]		

        return x, y

def create_dataloader(txt_file, tokenizer, batch_size=4, max_seq_len=256):
    dataset = CPPCodeDataset(txt_file, tokenizer, max_seq_len)

    if len(dataset) == 0:
        return None
 
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    return loader
