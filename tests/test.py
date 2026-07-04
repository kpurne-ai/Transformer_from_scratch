import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from model.model import Transformer, tokenizer, device, input_ids, embed_dim, num_heads, head_dim, vocab_size, max_seq_len, data

import torch
import torch.nn as nn
import torch.nn.functional as F



model = Transformer(embed_dim, num_heads, head_dim, vocab_size, max_seq_len).to(device)
model.load_state_dict(torch.load("transformer_model.pth"))
model.eval()
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

dataset = SimpleDataset(input_ids)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

class SimpleTrainer:
    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for batch in self.dataloader:
                batch = batch.to(device)
                output = self.model(batch)
                loss = criterion(output.view(-1, vocab_size), batch.view(-1))
                total_loss += loss.item()
        print(f"Evaluation Loss: {total_loss / len(self.dataloader)}")

trainer = SimpleTrainer(model, dataloader)
trainer.evaluate()
