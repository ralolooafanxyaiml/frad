import torch
import torch.nn.functional as F
from model import Transformer, ModelArgs
from tokenizer import CustomBPE
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "frad.pth"

def load_model():
    print(f"Downloading... Device: {DEVICE}")
   
    tokenizer = CustomBPE(vocab_size=500)
   
    args = ModelArgs(
        dim=512,
        n_layers=8,
        n_heads=8,
        vocab_size=4096,
        max_seq_len=256,
    )

    model = Transformer(args).to(DEVICE)
 
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Downloaded!")
    else:
        print("Error!")
    
    model.eval()
    return model, tokenizer

def generate(model, tokenizer, prompt, max_new_tokens=50, temperature=0.8):
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(DEVICE)
    print(f"\nPrompt: {prompt}")
    print("Generating...", end="", flush=True)

    for _ in range(max_new_tokens):
        idx_cond = tokens[:, -model.params.max_seq_len:]
  
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits / temperature

        probs = F.softmax(logits, dim=-1)
 
        idx_next = torch.multinomial(probs, num_samples=1)

        next_token_id = idx_next.item()
        decoded_word = tokenizer.id_to_token.get(next_token_id, b"")
        try:
            print(decoded_word.decode("utf-8", errors="ignore"), end="", flush=True)
        except:
            pass

        tokens = torch.cat((tokens, idx_next), dim=1)

    print("\n\nBitti!")
    return tokens

if __name__ == "__main__":
    model, tokenizer = load_model()

    user_input = "int main"
    generate(model, tokenizer, user_input)