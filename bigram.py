import torch
import torch.nn as nn
from torch.nn import functional as F
batch_size=32
block_size=8
max_iters=3000
eval_interval=300
learning_rate=1e-2
device='cuda' if torch.cuda.is_available() else 'cpu'
eval_iters=200
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



class BigramLanguageModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table=nn.Embedding(vocab_size,vocab_size)
    def forward(self,idx,targets=None): # idx is (B, T) array of indices in the current context
        logits = self.token_embedding_table(idx) # (B,T,C)
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
            logits, loss = self(idx)#basically first we are getting logits for the input sequence
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C) converts to probabilites that sum up to 100
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1) basically rolls a dice on those percentages
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
model = BigramLanguageModel(vocab_size)
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
context=torch.zeros((1,1),dtype=torch.long,device=device)
print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist()))