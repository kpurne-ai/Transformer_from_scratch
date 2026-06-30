import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# -----------------------------
# Hyperparameters
# -----------------------------
batch_size = 2
seq_len = 5
embed_dim = 8
num_heads = 2
head_dim = 4          

# Dummy Input
X = torch.randn(batch_size, seq_len, embed_dim)

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
        mask = torch.tril(torch.ones(seq_len, seq_len))
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
        self.ln_f = nn.LayerNorm(embed_dim)
    def forward(self, X):
        X = self.blocks(X)
        X = self.ln_f(X)
        return X
    
class Transformer(nn.Module):
    def __init__(self, embed_dim, num_heads, head_dim):
        super().__init__()
        self.blocks = TransformerBlocks(embed_dim, num_heads, head_dim)
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, embed_dim)
    def forward(self, X):
        X = self.blocks(X)
        X = self.ln_f(X)
        logits = self.lm_head(X)
        return logits   
    
print(X.shape)
model = Transformer(embed_dim, num_heads, head_dim)

output = model(X)
print(output.shape)
print(output)