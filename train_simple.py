from model.model import Transformer, tokenizer, device, input_ids, embed_dim, num_heads, head_dim, vocab_size, max_seq_len, data
import torch
import torch.nn as nn
import torch.optim as optim


model = Transformer(embed_dim, num_heads, head_dim, vocab_size, max_seq_len).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
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
    def __init__(self, model, optimizer, criterion, dataloader):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataloader = dataloader

    def train(self, num_epochs):
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in self.dataloader:
                batch = batch.to(device)
                self.optimizer.zero_grad()
                output = self.model(batch)
                loss = self.criterion(output.view(-1, vocab_size), batch.view(-1))
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(self.dataloader)}")
        
trainer = SimpleTrainer(model, optimizer, criterion, dataloader)
num_epochs = 10
trainer.train(num_epochs)

torch.save(model.state_dict(), "transformer_model.pth")
print("Model saved to transformer_model.pth")