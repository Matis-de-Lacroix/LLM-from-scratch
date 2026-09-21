"""
minigpt_dumas.py — a character-level GPT trained from scratch on Dumas.

A compact, decoder-only transformer (GPT-2 style) implemented by hand in PyTorch
and trained on Alexandre Dumas' "Les Trois Mousquetaires". Every component — the
character tokenizer, multi-head causal self-attention, the training loop and the
autoregressive sampler — is written explicitly rather than pulled from a library,
so the whole pipeline that produced the results fits in one readable file.

Run:
    pip install torch
    python minigpt_dumas.py

Trains in ~10-20 min on CPU (much faster on a GPU), then prints generated text at
three sampling temperatures.
"""
import os
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Hyperparameters
# --------------------------------------------------------------------------- #
BATCH_SIZE = 32          # sequences per training step
BLOCK_SIZE = 64          # context length (max tokens the model attends to)
N_EMBD     = 128         # embedding / residual stream width
N_HEAD     = 4           # attention heads (N_EMBD must be divisible by N_HEAD)
N_LAYER    = 4           # stacked transformer blocks
DROPOUT    = 0.1
MAX_STEPS  = 3000
LEARNING_RATE = 3e-4
EVAL_INTERVAL = 300      # steps between train/val loss estimates
EVAL_BATCHES  = 50       # batches averaged per loss estimate
SEED = 42

CORPUS_URL = "https://www.gutenberg.org/ebooks/13951.txt.utf-8"  # Les Trois Mousquetaires
device = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# 1. Corpus + character-level tokenizer
# --------------------------------------------------------------------------- #
def load_corpus(path="corpus.txt"):
    """Download the novel once and strip the Project Gutenberg header/footer."""
    if not os.path.exists(path):
        urllib.request.urlretrieve(CORPUS_URL, path)
    text = open(path, encoding="utf-8").read()
    start = text.find("***", text.find("*** START") + 3) + 3
    end = text.find("*** END")
    return text[start:end]


text = load_corpus()

# The tokenizer is deliberately trivial: one token = one character. This keeps
# the vocabulary tiny so that all the modelling effort goes into the transformer.
chars = sorted(set(text))
VOCAB_SIZE = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

# Encode the whole corpus and split 90/10 into train / validation.
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


def get_batch(split):
    """Sample a batch of (context, target) pairs.

    `y` is `x` shifted by one position: at every time step the target is the
    next character. This is the whole supervised signal behind "predict the
    next token".
    """
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([d[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([d[i + 1:i + 1 + BLOCK_SIZE] for i in ix])
    return x.to(device), y.to(device)


# --------------------------------------------------------------------------- #
# 2. The model
# --------------------------------------------------------------------------- #
class Block(nn.Module):
    """One transformer block: causal self-attention, then a feed-forward MLP.

    Both sub-layers use pre-normalisation (LayerNorm first) and a residual
    connection, exactly as in GPT-2. The residual paths are what let many blocks
    be stacked and still train.
    """

    def __init__(self):
        super().__init__()
        # Query, key and value projections are fused into a single Linear.
        self.qkv = nn.Linear(N_EMBD, 3 * N_EMBD, bias=False)
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.ffwd = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD), nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD), nn.Dropout(DROPOUT),
        )
        self.ln1, self.ln2 = nn.LayerNorm(N_EMBD), nn.LayerNorm(N_EMBD)
        self.drop = nn.Dropout(DROPOUT)
        # Causal mask: position t may only attend to positions <= t.
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

    def attention(self, x):
        B, T, C = x.shape
        hs = C // N_HEAD
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, N_HEAD, hs).transpose(1, 2)
        k = k.view(B, T, N_HEAD, hs).transpose(1, 2)
        v = v.view(B, T, N_HEAD, hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * hs ** -0.5           # scaled scores
        att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        att = self.drop(F.softmax(att, dim=-1))
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))   # communication: tokens exchange info
        x = x + self.ffwd(self.ln2(x))        # computation: each token thinks alone
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
        """Autoregressive sampling.

        temperature < 1 makes the model safer / more repetitive, > 1 more
        creative; top_k restricts sampling to the k most likely characters.
        """
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -BLOCK_SIZE:])       # crop to context window
            logits = logits[:, -1, :] / temperature      # last step only
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
        self.train()
        return idx


# --------------------------------------------------------------------------- #
# 3. Training
# --------------------------------------------------------------------------- #
@torch.no_grad()
def estimate_loss(model):
    """Average the loss over several batches of train and val — the gap between
    the two is our overfitting detector."""
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


# --------------------------------------------------------------------------- #
# 4. Run: train, then compare sampling temperatures
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    model = train()

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    for temp in (0.5, 1.0, 1.5):
        sample = decode(model.generate(context, 300, temperature=temp, top_k=40)[0].tolist())
        print(f"\n{'=' * 22} temperature = {temp} {'=' * 22}\n{sample}")
