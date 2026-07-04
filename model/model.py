import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

text = """
hello world.
this is my first transformer language model.
i am learning how tokenizers work.
the model reads text and predicts the next character.
transformers use attention to understand context.
hello again.
"""


# Character-level vocabulary
chars = sorted(list(set(text)))

# Token -> integer ID
stoi = {ch: i for i, ch in enumerate(chars)}

# Integer ID -> token
itos = {i: ch for ch, i in stoi.items()}

vocab_size = len(chars)

print("Vocabulary size:", vocab_size)
print("Vocabulary:", chars)


class CharTokenizer:
    def __init__(self, stoi, itos):
        self.stoi = stoi
        self.itos = itos

    def encode(self, text):
        
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):

        return "".join(self.itos[i] for i in ids)


tokenizer = CharTokenizer(stoi, itos)


data = torch.tensor(
    tokenizer.encode(text),
    dtype=torch.long
)

print("\nDataset tensor shape:", data.shape)
print("First token IDs:", data[:20])



# -----------------------------
# Hyperparameters
# -----------------------------
embed_dim = 8
num_heads = 2
head_dim = 4
max_seq_len = 512

device = torch.device("mps" if torch.mps.is_available() else "cpu")
input_ids = data.unsqueeze(0).to(device)

class Head(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()

        self.WQ = nn.Linear(embed_dim, head_dim, bias=False)
        self.WK = nn.Linear(embed_dim, head_dim, bias=False)
        self.WV = nn.Linear(embed_dim, head_dim, bias=False)

    def forward(self, X):
        Q = self.WQ(X)
        K = self.WK(X)
        V = self.WV(X)

        scores = (Q @ K.transpose(-2, -1)) / (head_dim ** 0.5)
        seq_len = X.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=X.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        output = weights @ V
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, head_dim):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(embed_dim, head_dim) for _ in range(num_heads)]
        )
        self.WO = nn.Linear(num_heads * head_dim, embed_dim, bias=False)
    def forward(self, X):
        head_outputs = [head(X) for head in self.heads]
        concat = torch.cat(head_outputs, dim=-1)
        output = self.WO(concat)
        return output


class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
    def forward(self, X):
        return self.net(X)



class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, head_dim):
        super().__init__()
        self.mha = MultiHeadAttention(embed_dim, num_heads, head_dim)
        self.ff = FeedForward(embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        
    def forward(self, X):
        X = X + self.mha(self.ln1(X))
        X = X + self.ff(self.ln2(X))
        return X
    
class TransformerBlocks(nn.Module):
    def __init__(self, embed_dim, num_heads, head_dim):
        super().__init__()
        self.blocks = nn.Sequential(*[Block(embed_dim, num_heads, head_dim) for _ in range(10)])

    def forward(self, X):
        return self.blocks(X)
    
class Transformer(nn.Module):
    def __init__(self, embed_dim, num_heads, head_dim, vocab_size, max_seq_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.blocks = TransformerBlocks(embed_dim, num_heads, head_dim)
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(input_ids.size(0), -1)
        X = self.embedding(input_ids) + self.pos_embedding(position_ids)
        X = self.blocks(X)
        X = self.ln_f(X)
        logits = self.lm_head(X)
        return logits

print("Input IDs shape:", input_ids.shape)
model = Transformer(embed_dim, num_heads, head_dim, vocab_size, max_seq_len).to(device)

output = model(input_ids)
print("Output shape:", output.shape)
print(output)