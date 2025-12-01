import torch
import torch.nn as nn
import torch.optim as optim
import os

from model import Transformer, ModelArgs
from tokenizer import CustomBPE
from data import create_dataloader

BATCH_SIZE = 8
MAX_SEQ_LEN = 512
LEARNING_RATE = 3e-4
EPOCHS = 10
VOCAB_SIZE = 4096
INPUT_FILE = "input_code.txt"
SAVEE_PATH = "frad.pth"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using: {device.upper()}")

    if not os.path.exists(INPUT_FILE):
        return

    print("Tokenizer is studying...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        text_data = f.read()

    tokenizer = CustomBPE(vocab_size=VOCAB_SIZE)
    tokenizer.train(text_data)
    print("Tokenizer is ready!")

    train_loader = create_dataloader(INPUT_FILE, tokenizer, BATCH_SIZE, MAX_SEQ_LEN)
    if train_loader is None:
        print("Error!")
        return

    print("Model is building itself.")
    args = ModelArgs(
        dim=512,
        n_layers=8,
        n_heads=8,
        vocab_size=tokenizer.vocab_size,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=BATCH_SIZE
    )
    model = Transformer(args).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
  
    criterion = nn.CrossEntropyLoss()

    model.train()
    print("Study time.")

    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)

            loss = criterion(logits.view(-1, tokenizer.vocab_size), y.view(-1))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
 
            if batch_idx % 10 == 0:
               print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} is done. Average Loss is: {avg_loss:.4f}")

    torch.save(model.state_dict(), "frad.pth")

    
