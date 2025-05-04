import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from spikingjelly.activation_based import neuron, encoding, functional, layer
import os
import matplotlib.pyplot as plt
import csv
from pathlib import Path

# ==== CONFIGURATION ====
BATCH_SIZE = 64
EPOCHS = 5
TIME_STEPS = 10
LEARNING_RATE = 1e-3
QUANTIZE_WEIGHTS=True
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
def quantize_model_weights(model, num_bits):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name:
                w_min = param.min()
                w_max = param.max()
                delta = (w_max - w_min) / (2 ** num_bits - 1)
                param.copy_(torch.round((param - w_min) / delta) * delta + w_min)

# ==== ACCURACY CALCULATION ====
@torch.no_grad()
def calculate_accuracy(model, data_loader, encoder):
    model.eval()
    correct = 0
    total = 0
    for images, labels in data_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        spike_input = images.permute(1, 0, 2, 3, 4)  # [B, T, C, H, W] → [T, B, C, H, W]
        outputs = model(spike_input)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        functional.reset_net(model)
    return 100 * correct / total

class PreEncodedPoissonDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, encoder):
        self.encoded_data = []
        for image, label in base_dataset:
            image = image.unsqueeze(0)  # [1, H, W]
            spikes = encoder(image).float()  # [T, 1, 1, H, W]
            if spikes.shape[2] == 1:
                spikes = spikes.squeeze(2)  # Remove channel dim if present
                self.encoded_data.append((spikes, label))

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        return self.encoded_data[idx]


# ==== MAIN FUNCTION ====
def main():
    image_sizes = [28] #[4, 7, 14, 28]
    quant_bits_list = [2, 3, 4, 5]
    time_step_list = [20] #[5, 10, 20]

    for img_size in image_sizes:
        for time_steps in time_step_list:
            print(f"\n=== IMG: {img_size}x{img_size}, TIME_STEPS: {time_steps} ===")

            global IMG_SIZE, TIME_STEPS
            IMG_SIZE = img_size
            TIME_STEPS = time_steps

            # Dataset and Transform
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor()
            ])
            train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
            # Initialize encoder before encoding
            encoder = PoissonEncoder(time_steps=TIME_STEPS)

            # Pre-encode entire dataset
            print("Pre-encoding dataset...")
            train_dataset = PreEncodedPoissonDataset(train_dataset, encoder)
            test_dataset = PreEncodedPoissonDataset(test_dataset, encoder)
            print("Pre-encoding complete.")

            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


            # Model + Encoder
            encoder = PoissonEncoder(time_steps=TIME_STEPS)
            model = SNN().to(DEVICE)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

            # Training
            for epoch in range(EPOCHS):
                model.train()
                running_loss = 0.0
                for images, labels in train_loader:
                    images = images.to(DEVICE)
                    labels = labels.to(DEVICE)
                    spike_input = images.permute(1, 0, 2, 3, 4)  # [B, T, C, H, W] → [T, B, C, H, W]
                    outputs = model(spike_input)
                    loss = criterion(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    functional.reset_net(model)
                    running_loss += loss.item()

                train_acc = calculate_accuracy(model, train_loader, encoder)
                test_acc = calculate_accuracy(model, test_loader, encoder)
                print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

            # Post-Training Quantization
            if QUANTIZE_WEIGHTS:
                for bits in quant_bits_list:
                    model_copy = SNN().to(DEVICE)
                    model_copy.load_state_dict(model.state_dict())
                    quantize_model_weights(model_copy, bits)
                    test_acc_q = calculate_accuracy(model_copy, test_loader, encoder)
                    print(f"[Quantized @ {bits}-bit] Test Accuracy: {test_acc_q:.2f}%")

                    # === CSV Logging ===
                    with open("snn_results.csv", mode="a", newline="") as file:
                        writer = csv.DictWriter(file, fieldnames=["img_size", "time_steps", "quant_bits", "test_accuracy"])
                        writer.writerow({
                            "img_size": img_size,
                            "time_steps": time_steps,
                            "quant_bits": bits,
                            "test_accuracy": test_acc_q
                        })

            print(f"=== Done: IMG {img_size} | TIME_STEPS {time_steps} ===\n")
            torch.save(model.state_dict(), "trained_supervised_snn.pt")  # Save trained model after last loop


if __name__ == '__main__':
    main()



# ==== RASTER PLOT SECTION ====
import numpy as np


def plot_raster(spike_record, title, save_path):
    plt.figure(figsize=(10, 4))
    raster_data = []
    for t in range(spike_record.shape[0]):
        fired = spike_record[t].nonzero(as_tuple=True)[0]
        if len(fired) == 0:
            continue
        for i in fired:
            raster_data.append((t * 1e-6, int(i)))
        plt.scatter([t] * len(fired), fired, s=2)
    plt.title(title)
    plt.xlabel("Time (step)")
    plt.ylabel("Neuron Index")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    # Save to .txt in time (s) format
    with open(save_path.replace('.png', '.txt'), 'w') as f:
        for t, i in raster_data:
            f.write(f"{t:.1e}\t{i}\n")



# Use final trained model with IMG_SIZE=28, TIME_STEPS=20
IMG_SIZE = 28
TIME_STEPS = 20
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])
test_dataset_full = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
encoder = PoissonEncoder(time_steps=TIME_STEPS)
model = SNN().to(DEVICE)
model.load_state_dict(torch.load("trained_supervised_snn.pt", weights_only=True))  # assumes you save the trained model
model.eval()

# Collect samples for digits [0, 6, 8]
target_digits = [0, 6, 8]
samples = {d: None for d in target_digits}
for img, label in test_dataset_full:
    if label in target_digits and samples[label] is None:
        samples[label] = (img.unsqueeze(0).to(DEVICE), label)
    if all(v is not None for v in samples.values()):
        break

# Redefine SNN with spike logging
class SNNWithSpikes(SNN):
    def forward(self, x_spike):
        input_spikes = []
        hidden_spikes = []
        output_spikes = []
        for t in range(x_spike.shape[0]):
            x = x_spike[t].view(x_spike.size(1), -1)
            input_spikes.append(x.clone().detach().cpu())
            x = self.fc1(x)
            x = self.neuron1(x)
            hidden_spikes.append(x.clone().detach().cpu())
            x = self.fc2(x)
            x = self.neuron2(x)
            output_spikes.append(x.clone().detach().cpu())
        return torch.stack(input_spikes), torch.stack(hidden_spikes), torch.stack(output_spikes)

model_spike = SNNWithSpikes().to(DEVICE)
model_spike.load_state_dict(model.state_dict())

# Generate raster plots
for digit, (img, lbl) in samples.items():
    spikes = encoder(img).float().to(DEVICE)  # [T, B, C, H, W]
    spikes = spikes.view(TIME_STEPS, 1, -1)   # flatten
    in_spk, hid_spk, out_spk = model_spike(spikes)

    plot_raster(in_spk.squeeze().T, f"Input Spikes - Digit {digit}", f"raster_input_{digit}.png")
    plot_raster(hid_spk.squeeze().T, f"Hidden Spikes - Digit {digit}", f"raster_hidden_{digit}.png")
    plot_raster(out_spk.squeeze().T, f"Output Spikes - Digit {digit}", f"raster_output_{digit}.png")
