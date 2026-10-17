# ==========================================
# Section 1: Environment Setup & Device Check
# ==========================================
# Checks if PyTorch is installed correctly and configures 
# the hardware accelerator (GPU/CUDA if available, otherwise CPU).
import torch
print("Using PyTorch version:", torch.__version__)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)


# ==========================================
# Section 2: Dataset Download & Loading
# ==========================================
# Downloads Andrej Karpathy's Tiny Shakespeare text corpus 
# and reads the raw text contents into memory for processing.
import urllib.request

# Download tiny shakespeare dataset
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
urllib.request.urlretrieve(url, "input.txt")

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("Dataset length (characters):", len(text))


# ==========================================
# Section 3: Tokenization & Vocabulary Mapping
# ==========================================
# Extracts all unique characters from the text to create a character-level 
# vocabulary. Defines encoder and decoder functions to map text strings to integer tokens and vice versa.
# Create a vocabulary of unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("Vocab size:", vocab_size)

# Create mappings from characters to integers and vice versa
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
print("stoi:", stoi)
print("itos:", itos)

encode = lambda s: [stoi[c] for c in s] # encoder: take string, output list of ints
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take list of ints, output string

# Test our encoder and decoder
print(encode("hello world"))
print(decode(encode("hello world")))


# ==========================================
# Section 4: Data Batching & Train/Val Split
# ==========================================
# Encodes the complete text dataset into a long PyTorch tensor, splits it into 
# 90% training and 10% validation sets, and builds a batch generator function 
# to sample random chunks of data (contexts and target predictions).
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Encode the entire text into a tensor
data = torch.tensor(encode(text), dtype=torch.long)

# Split into train and validation sets
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
batch_size = 64
block_size = 256 # Context length for predictions

def get_batch(split):
    data = train_data if split == 'train' else val_data
    # Generate random starting indices for batches
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# Test getting a batch and printing shapes
xb, yb = get_batch('train')
print("inputs shape:", xb.shape)
print("targets shape:", yb.shape)
print("target example (decoded):", decode(yb[0].tolist()[:50]))


# ==========================================
# Section 5: Transformer Architecture Definition
# ==========================================
# Defines the full neural network architecture: Self-Attention Heads, 
# Multi-Head Attention, Feed-Forward layers, Residual Transformer Blocks, 
# and the complete TinyLanguageModel incorporating token and position embeddings.
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    """ One head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # Compute attention scores
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

class FeedForward(nn.Module):
    """ A simple linear layer followed by a non-linearity """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class TinyLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Hyperparameters
n_embd = 384
n_head = 6
n_layer = 6
learning_rate = 3e-4

model = TinyLanguageModel().to(device)
print("Total parameters:", sum(p.numel() for p in model.parameters())/1e3, "k")


# ==========================================
# Section 6: Model Training Loop
# ==========================================
# Initializes the AdamW optimizer and runs the training loop over multiple 
# iterations, computing loss, backpropagating gradients, and updating weights.
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
max_iters = 5000

for iter in range(max_iters):
    xb, yb = get_batch('train')
    
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    if iter % 100 == 0:
        print(f"Iteration {iter}, Loss: {loss.item():.4f}")


# ==========================================
# Section 7: Text Generation
# ==========================================
# Uses the trained model starting from an empty context to generate 
# and decode novel Shakespeare-like text token by token.
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text = decode(model.generate(context, max_new_tokens=500)[0].tolist())
print("--- Generated Text ---")
print(generated_text)


# ==========================================
# Section 8: Saving Model Weights
# ==========================================
# Serializes and saves the trained model's state dictionary to disk, 
# and triggers an automatic download to your local machine.
import torch
from google.colab import files

# Save the model's learned weights (state dictionary)
torch.save(model.state_dict(), 'tiny_shakespeare_llm.pt')
print("Model successfully saved as tiny_shakespeare_llm.pt")

# Automatically download the file to your local computer
files.download('tiny_shakespeare_llm.pt')


# ==========================================
# Section 9: Loading & Evaluating Saved Model
# ==========================================
# Re-instantiates the model architecture and loads the saved model 
# weights from the checkpoint file for evaluation or inference.
# Initialize the model architecture with the same hyperparameters
model = TinyLanguageModel().to(device)

# Load the saved state dictionary
model.load_state_dict(torch.load('tiny_shakespeare_llm.pt', map_location=device))

# Put the model in evaluation mode
model.eval()

print("Model successfully loaded and ready for text generation!")