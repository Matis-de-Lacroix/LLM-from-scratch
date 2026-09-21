"""
A small character-level GPT, written from scratch and trained on Dumas'
"Les Trois Mousquetaires".

Decoder-only transformer, GPT-2 style. Nothing here comes from a modelling
library: the tokenizer, causal attention, training loop and sampler are all
spelled out. ~830k parameters, small enough to train on a laptop.

    pip install torch
    python minigpt_dumas.py

Roughly 15 min on CPU, a couple of minutes on a GPU. Prints samples at three
temperatures once it's done.
"""
import os
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- hyperparameters ---------------------------------------------------------
BATCH_SIZE = 32
BLOCK_SIZE = 64          # context window, and therefore the size of pos_emb
N_EMBD     = 128
N_HEAD     = 4           # has to divide N_EMBD
N_LAYER    = 4
DROPOUT    = 0.1
MAX_STEPS  = 3000
LEARNING_RATE = 3e-4
EVAL_INTERVAL = 300
EVAL_BATCHES  = 50       # one batch is far too noisy to judge progress on
SEED = 42

CORPUS_URL = "https://www.gutenberg.org/ebooks/13951.txt.utf-8"  # Les Trois Mousquetaires
device = "cuda" if torch.cuda.is_available() else "cpu"


# --- corpus and tokenizer ----------------------------------------------------
def load_corpus(path="corpus.txt"):
    """Fetch the novel once, then cut off the Gutenberg header and licence."""
    if not os.path.exists(path):
        urllib.request.urlretrieve(CORPUS_URL, path)
    text = open(path, encoding="utf-8").read()
    # the banner reads "*** START OF ... ***", so jump to its closing *** —
    # stopping at the first one leaves the banner text in the corpus
    start = text.find("***", text.find("*** START") + 3) + 3
    end = text.find("*** END")
    return text[start:end]


text = load_corpus()

# one token = one character. Crude next to BPE, but it keeps the vocab around a
# hundred entries and puts all the difficulty where it's interesting.
chars = sorted(set(text))
VOCAB_SIZE = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

# 90/10 split, validation held out from the end of the book
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


def get_batch(split):
    """Random (context, target) pairs, both (B, T).

    y is x shifted one character to the right — that shift is the whole
    supervision signal behind "predict the next token".
    """
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([d[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([d[i + 1:i + 1 + BLOCK_SIZE] for i in ix])
    return x.to(device), y.to(device)


# --- model -------------------------------------------------------------------
class Block(nn.Module):
    """Causal self-attention, then an MLP, pre-norm and residual on both.

    Same layout as GPT-2. The residual path is the load-bearing part: strip it
    out and a stack this deep stops training properly.
    """

    def __init__(self):
        super().__init__()
        # q, k and v in one matmul instead of three — same maths, less overhead
        self.qkv = nn.Linear(N_EMBD, 3 * N_EMBD, bias=False)
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.ffwd = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD), nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD), nn.Dropout(DROPOUT),
        )
        self.ln1, self.ln2 = nn.LayerNorm(N_EMBD), nn.LayerNorm(N_EMBD)
        self.drop = nn.Dropout(DROPOUT)
        # lower triangle: token t may only look at 0..t, never at the future
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

    def attention(self, x):
        B, T, C = x.shape
        hs = C // N_HEAD
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, N_HEAD, hs).transpose(1, 2)           # (B, nh, T, hs)
        k = k.view(B, T, N_HEAD, hs).transpose(1, 2)
        v = v.view(B, T, N_HEAD, hs).transpose(1, 2)
        # the 1/sqrt(hs) matters: without it the dot products grow with head
        # size and softmax saturates into something close to one-hot
        att = (q @ k.transpose(-2, -1)) * hs ** -0.5           # (B, nh, T, T)
        att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        att = self.drop(F.softmax(att, dim=-1))
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)   # heads back together
        return self.proj(out)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))   # tokens look at each other
        x = x + self.ffwd(self.ln2(x))        # then each one digests on its own
        return x


class GPT(nn.Module):
    """Token + position embeddings -> N blocks -> final LayerNorm -> LM head."""

    def __init__(self, vocab_size):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, N_EMBD)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        logits = self.lm_head(self.ln_f(self.blocks(x)))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Sample one character at a time, feeding each one back in.

        temperature < 1 sharpens the distribution (safe, repetitive), > 1
        flattens it (wilder). top_k throws away everything outside the k best
        candidates, which is what stops the odd absurd character slipping in.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # pos_emb only knows BLOCK_SIZE positions, so keep the last ones
            logits, _ = self(idx[:, -BLOCK_SIZE:])
            logits = logits[:, -1, :] / temperature      # only the next step matters
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
        self.train()
        return idx


# --- training ----------------------------------------------------------------
@torch.no_grad()
def estimate_loss(model):
    """Loss averaged over a few batches of each split.

    A single batch bounces around far too much to tell whether anything is
    improving. The gap between the two numbers is the overfitting signal.
    """
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(EVAL_BATCHES)
        for i in range(EVAL_BATCHES):
            _, loss = model(*get_batch(split))
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train():
    torch.manual_seed(SEED)
    model = GPT(VOCAB_SIZE).to(device)
    print(f"device: {device} | vocab: {VOCAB_SIZE} | "
          f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    for step in range(MAX_STEPS + 1):
        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % EVAL_INTERVAL == 0:
            l = estimate_loss(model)
            print(f"step {step:4d} | train {l['train']:.3f} | val {l['val']:.3f}")
    return model


# --- train, then listen to the thing at three temperatures -------------------
if __name__ == "__main__":
    model = train()

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    for temp in (0.5, 1.0, 1.5):
        sample = decode(model.generate(context, 300, temperature=temp, top_k=40)[0].tolist())
        print(f"\n{'=' * 22} temperature = {temp} {'=' * 22}\n{sample}")
