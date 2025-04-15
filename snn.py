import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from spikingjelly.activation_based import neuron, encoding, functional, layer

import os
import matplotlib.pyplot as plt

# ==== CONFIGURATION ====
BATCH_SIZE = 64
EPOCHS = 5
TIME_STEPS = 10
LEARNING_RATE = 1e-3

IMG_SIZE = 28
HIDDEN_NEURONS = 512
OUTPUT_NEURONS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== POISSON ENCODER ====
class PoissonEncoder(torch.nn.Module):
    def __init__(self, time_steps):
        super().__init__()
        self.time_steps = time_steps

    def forward(self, x):
        # Input shape: [batch, channels, height, width]
        # Output shape: [time_steps, batch, channels, height, width]
        return torch.rand((self.time_steps,) + x.shape, device=x.device) < x

# ==== SNN DEFINITION ====
class SNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = layer.Linear(IMG_SIZE*IMG_SIZE, HIDDEN_NEURONS)
        self.neuron1 = neuron.IFNode()
        self.fc2 = layer.Linear(HIDDEN_NEURONS, OUTPUT_NEURONS) # 512 neuron layers, 10 output layers
        self.neuron2 = neuron.IFNode()

    def forward(self, x_spike):
        mem_rec = []
        for t in range(x_spike.shape[0]):
            x = x_spike[t].view(x_spike.size(1), -1)  # Flatten: [B, 28*28]
            x = self.fc1(x)
            x = self.neuron1(x)
            x = self.fc2(x)
            x = self.neuron2(x)
            mem_rec.append(x)
        return torch.stack(mem_rec).mean(0)

# ==== WEIGHT QUANTIZATION ==== 


# ==== ACCURACY CALCULATION ====
@torch.no_grad()
def calculate_accuracy(model, data_loader, encoder):
    model.eval()
    correct = 0
    total = 0
    for images, labels in data_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        spike_input = encoder(images).float()
        outputs = model(spike_input)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        functional.reset_net(model)
    return 100 * correct / total

# ==== MAIN FUNCTION ====
def main():
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    encoder = PoissonEncoder(time_steps=TIME_STEPS)     # Poisson Spike Encoding called here
    model = SNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            spike_input = encoder(images.float()).float()
            outputs = model(spike_input)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()     # Surrogate Gradient Descent (supervised learning)
            optimizer.step()
            functional.reset_net(model)
            running_loss += loss.item()

        # Calculate and print accuracies
        train_acc = calculate_accuracy(model, train_loader, encoder)
        test_acc = calculate_accuracy(model, test_loader, encoder)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    print("Training complete.")

if __name__ == '__main__':
    main()
