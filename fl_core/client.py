import torch
import torch.nn as nn
import torch.optim as optim

class FLClient:
    def __init__(self, model, data_loader, device='cpu'):
        self.model = model.to(device)
        self.data_loader = data_loader
        self.device = device

    def train(self, epochs=1, lr=0.01):
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        for _ in range(epochs):
            for data, target in self.data_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        return self.model.state_dict()
