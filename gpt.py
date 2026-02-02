import torch
import torch.nn as nn
import torch.nn.functional as F


# Hyperparameters
batch_size = 8 # 
block_size = 256 # Maximum context length for prediction
train_steps = 100000
eval_steps = 200
learning_rate = 3e-4
dropout = 0.2
num_heads = 6
num_layers = 6
device = "cuda" if torch.cuda.is_available() else 'cpu'
n_embd = 384


torch.manual_seed(1337)

with open('input.txt', 'r') as f:
    text = f.read()

print("length of dataset in characters: ", len(text))

chars = sorted(list(set(text)))
vocab_size = len(chars)

print("vocab_size: ",vocab_size)
print("vocab: ",''.join(chars))

stoi = {ch : i for i, ch in enumerate(chars)}
itos = {i : ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(encode("Hi there!"))
print(decode(encode("Hi there!")))

data = torch.tensor(encode(text), dtype=torch.long)

print(data.shape, data.dtype)
print(data[:10])

n = int(0.9*len(data))

train_data = data[:n]
val_data = data[n:]

x = train_data[:block_size]
y = train_data[1:block_size+1]

for t in range(block_size):
    input = x[:t+1]
    target = y[t]
    print(f"When input is: {input}, the target is: {target}")

def get_batch(batch_size, split):
    data = train_data if split == "train" else val_data
    idx = torch.randint(len(data) - block_size, (batch_size,))
    input = torch.stack([data[i:i+block_size] for i in idx])
    target = torch.stack([data[i+1:i+block_size + 1] for i in idx])

    return input.to(device), target.to(device)

x, y = get_batch(batch_size, "train")
print('inputs:')
print("shape", x.shape)
print(x)

print('outputs:')
print("shape", y.shape)
print(y)

for b in range(batch_size):
    for i in range(block_size):
        input = x[b, :i+1]
        target = y[b, i]
        print(f"When input is: {input.tolist()}, the target is: {target}")

@torch.no_grad()
def evaluate(model):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = []
        for iters in range(eval_steps):
            x, y = get_batch(batch_size, split)
            logits, loss = model(x, y)
            losses.append(loss.item())
        out[split]  = sum(losses)/len(losses)
    model.train()
    return out
        
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("trill", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        wei = q @ k.transpose(-2, -1) * C**0.5
        wei = wei.masked_fill(self.trill[:T, :T]==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj_out = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj_out(out))
        return out
    
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4*n_embd),
                                 nn.ReLU(),
                                 nn.Linear(4*n_embd, n_embd), # Acts as Projection Layer
                                 nn.Dropout(dropout)
                                 )
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, num_heads, n_embd):
        super().__init__()
        self.sa_head = MultiHeadAttention(num_heads, n_embd//num_heads)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[Block(num_heads, n_embd) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, target=None):

        # idx = [B , T]
        # target = [B, T]
        B, T = idx.shape
        
        tok_emb = self.token_embedding_table(idx) # [B, T, n_embd]
        pos_emb = self.position_embedding_table(torch.arange(T, device = device)) # [T, n_embd]
        x = tok_emb + pos_emb
        
        x = self.blocks(x)
        x = self.ln_f(x)
            
        logits = self.lm_head(x) # [B, T, vocab_size]
        
        if target is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            target = target.view(B*T)
            loss = F.cross_entropy(logits, target)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx = [B, T]array of indices in the current context

        for _ in range(max_new_tokens):
            # crp the index till block_size
            idx_cond = idx[:, -block_size:]
            # get the prediction
            logits, _ = self(idx_cond) 
            # Focus on last time step
            logits = logits[:,-1,:] # [B, C]
            # apply softmax to get the probabilty
            probs = F.softmax(logits, dim=-1)
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1) # [B, 1]
            # append sampled index to the running sequnce
            idx = torch.cat((idx, idx_next), dim=1) # [B, T+1]
        return idx
    
model = GPTLanguageModel().to(device)
logits, loss = model(x, y)
print(logits.shape, loss)

decode(model.generate(torch.tensor(encode("Hi there"), dtype=torch.long).unsqueeze(0).to(device), max_new_tokens=100)[0].tolist())

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


for step in range(train_steps):
    x, y = get_batch(batch_size, "train")
    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step%eval_steps==0:
        out = evaluate(model)
        print(f"Step-{step}:- Train Loss is {out['train']} and Val Loss is {out['val']}")
        generated_text = decode(model.generate(torch.tensor(encode("Hi there"), dtype=torch.long).unsqueeze(0).to(device), max_new_tokens=100)[0].tolist())
        print()
        print("generated_text:- ", generated_text)
        print()