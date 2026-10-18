import torch
import torch.nn as nn
from torch.nn import functional as F
batch_size=32
block_size=8
max_iters=5000
eval_interval=500
learning_rate=1e-3
device='cuda' if torch.cuda.is_available() else 'cpu'
eval_iters=200
n_embd=384
n_head=6
n_layer=6
dropout=0.2
torch.manual_seed(1337)


import urllib.request
import os

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
file_path = "input.txt"

if not os.path.exists(file_path):
    print("Downloading dataset...")
    urllib.request.urlretrieve(url, file_path)
    print("Download complete.")
else:
    print("Dataset already exists.")


with open('input.txt',encoding='utf-8') as f:
    text=f.read()


chars=sorted(list(set(text)))
vocab_size=len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string



data=torch.tensor(encode(text),dtype=torch.long)


n=int(0.9*len(data))
train_data=data[:n]
val_data=data[n:]



def get_batch(split):
    if(split=='train'):
        data=train_data
    else :
        data=val_data
    ix=torch.randint(len(data)-block_size,(batch_size,)) #basically generates 4 random offsets/integers between the given range
    x=torch.stack([data[i:i+block_size] for i in ix]) # so this creates a 4x8 tensor where the 8 columns we are generating then stacking
    y=torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y=x.to(device),y.to(device)
    return x,y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
class Head(nn.Module):
    """one head self-attention"""
    def __init__(self, head_size):
        super().__init__()
        self.key=nn.Linear(n_embd,head_size,bias=False)
        self.query=nn.Linear(n_embd,head_size,bias=False)
        self.value=nn.Linear(n_embd,head_size,bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        B,T,C=x.shape
        k=self.key(x)
        q=self.query(x)
        v=self.value(x) # B,T,C
        # k.shape[-1] automatically grabs the correct head_size
        wei = q @ k.transpose(-2,-1) * (k.shape[-1]**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei=F.softmax(wei,dim=-1)
        wei=self.dropout(wei)
        out=wei @ v # B,T,T B,T,C ---> B,T,C
        return out
class MultiHeadAttention(nn.Module):
    """multiple heads of self attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads=nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj=nn.Linear(n_embd,n_embd)
        self.dropout=nn.Dropout(dropout)
    def forward(self ,x):
        out=torch.cat([h(x) for h in self.heads],dim=-1)
        out=self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4* n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd,n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table=nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table=nn.Embedding(block_size,n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head=nn.Linear(n_embd,vocab_size)
    def forward(self,idx,targets=None): # idx is (B, T) array of indices in the current context
        B,T=idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb=self.position_embedding_table(torch.arange(T,device=device))# basically from 0 to t-1
        x=tok_emb+pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits=self.lm_head(x) #B,T,vocab_size
        #idx is xb
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)#basically flattening
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            idx_cond=idx[:,-block_size:]
            logits, loss = self(idx_cond)#basically first we are getting logits for the input sequence
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C) converts to probabilites that sum up to 100
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1) basically rolls a dice on those percentages
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
if __name__ == '__main__':
    model = BigramLanguageModel()
    m=model.to(device)
    optimizer=torch.optim.AdamW(m.parameters(),lr=1e-3)
    for iter  in range(max_iters):
        if(iter%eval_interval==0):
            losses=estimate_loss()
            print( f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train')

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    context = torch.zeros((1,1), dtype=torch.long, device=device)
    print(decode(m.generate(idx=context, max_new_tokens=500)[0].tolist()))