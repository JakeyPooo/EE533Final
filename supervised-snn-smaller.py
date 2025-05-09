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
import numpy as np
import matplotlib.cm as cm

# ==== CONFIGURATION ====
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-3
QUANTIZE_WEIGHTS = True
HIDDEN_NEURONS = 5
OUTPUT_NEURONS = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== POISSON ENCODER ====
class PoissonEncoder(torch.nn.Module):
    def __init__(self, time_steps):
        super().__init__()
        self.time_steps = time_steps

    def forward(self, x):
        return torch.rand((self.time_steps,) + x.shape, device=x.device) < x

# ==== SNN DEFINITION ====
class SNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = layer.Linear(IMG_SIZE * IMG_SIZE, OUTPUT_NEURONS)
        self.neuron1 = neuron.IFNode()

    def forward(self, x_spike):
        out = 0
        for t in range(x_spike.shape[0]):
            x = x_spike[t].view(x_spike.size(1), -1)
            x = self.fc1(x)
            x = self.neuron1(x)
            out += x
        return out / x_spike.shape[0]

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
        spike_input = images.permute(1, 0, 2, 3, 4)
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
            image = image.unsqueeze(0)
            spikes = encoder(image).float()
            if spikes.shape[2] == 1:
                spikes = spikes.squeeze(2)
                self.encoded_data.append((spikes, label))

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        return self.encoded_data[idx]

def plot_weight_histogram(model, bits, save_path):
    all_weights = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            all_weights.append(param.detach().cpu().numpy().flatten())
    all_weights = np.concatenate(all_weights)

    plt.figure(figsize=(8, 5))
    plt.hist(all_weights, bins=40, color='cornflowerblue', edgecolor='black')
    plt.title(f"Weight Distribution (Quantized to {bits}-bit)")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved weight histogram: {save_path}")

def save_output_spike_txt(output_spike_tensor, digit, save_dir="./plots", time_step_us=0.1):
    output_spike_tensor = output_spike_tensor.cpu().numpy()
    flags = (output_spike_tensor.sum(axis=0) > 0).astype(int)

    os.makedirs(save_dir, exist_ok=True)
    file_path = f"{save_dir}/output_digit{digit}.txt"

    with open(file_path, 'w') as f:
        for t, flag in enumerate(flags):
            f.write(f"{t * time_step_us:.1f}\t{flag}\n")

    print(f"Saved output spike .txt for digit {digit} → {file_path}")


# ==== MAIN FUNCTION ====
def main():
    image_sizes = [28]#[4, 7, 14, 28]
    quant_bits_list = [2, 3, 4, 5]
    time_step_list = [20]#[5, 10, 20]

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
            torch.save(model.state_dict(), "trained_supervised_snn_smaller.pt")  # Save trained model after last loop

            # Create directory if needed
            os.makedirs('./plots', exist_ok=True)

            # === Plot histogram for each quantization level ===
            for bits in [2, 3, 4, 5]:
                model_copy = SNN().to(DEVICE)
                model_copy.load_state_dict(model.state_dict())  # Load full-precision trained model
                quantize_model_weights(model_copy, bits)
                plot_weight_histogram(model_copy, bits=bits, save_path=f"./plots/smaller_weight_histogram_{bits}bit.png")
                

if __name__ == '__main__':
    main()

# ==== RASTER PLOT SECTION ====
def plot_combined_raster(input_spk, hidden_spk, output_spk, digit_label, save_path):
    """
    Plots input, hidden, and output spikes in one figure as subplots
    All input shapes must be [neurons, time]
    """
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    titles = ['Input Spikes', 'Hidden Spikes', 'Output Spikes']
    spike_sets = [input_spk, hidden_spk, output_spk]

    for ax, spike_tensor, title in zip(axs, spike_sets, titles):
        spike_tensor = spike_tensor.cpu().numpy()
        spikes_by_neuron = [np.where(spike_tensor[i] > 0)[0] for i in range(spike_tensor.shape[0])]

        # Optional: color map by neuron index
        colors = cm.get_cmap('viridis')(np.linspace(0, 1, len(spikes_by_neuron)))
        ax.eventplot(spikes_by_neuron, colors=colors, lineoffsets=1, linelengths=0.8)
        ax.set_title(f"{title} - Digit {digit_label}")
        ax.set_ylabel("Neuron Index")
        ax.set_ylim(-0.5, len(spikes_by_neuron) - 0.5)
        ax.grid(True, linestyle=':', linewidth=0.5)

        # Print spike rate stats
        firing_rates = [len(s) / TIME_STEPS for s in spikes_by_neuron]
        print(f"{title} Avg Firing Rate (Digit {digit_label}): {np.mean(firing_rates):.2f} spikes/time step")

    axs[-1].set_xlabel("Time step")
    axs[-1].set_xlim(0, TIME_STEPS)  # Optional: change to (0, 10) to zoom

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# === Sample digit plotting ===
IMG_SIZE = 28
TIME_STEPS = 20
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])
test_dataset_full = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
encoder = PoissonEncoder(time_steps=TIME_STEPS)
model = SNN().to(DEVICE)
model.load_state_dict(torch.load("trained_supervised_snn_smaller.pt", map_location=DEVICE))
model.eval()

target_digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # changes how many digits are ran
samples = {d: None for d in target_digits}
for img, label in test_dataset_full:
    if label in target_digits and samples[label] is None:
        samples[label] = (img.unsqueeze(0).to(DEVICE), label)
    if all(v is not None for v in samples.values()):
        break

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
            # x = self.fc2(x)
            # x = self.neuron2(x)
            output_spikes.append(x.clone().detach().cpu())
        return torch.stack(input_spikes), torch.stack(hidden_spikes), torch.stack(output_spikes)

model_spike = SNNWithSpikes().to(DEVICE)
model_spike.load_state_dict(model.state_dict())

for digit, (img, lbl) in samples.items():
    spikes = encoder(img).float().to(DEVICE)  # [T, B, 1, H, W]
    spikes_flat = spikes.view(TIME_STEPS, 1, -1)  # Flatten

    in_spk, hid_spk, out_spk = model_spike(spikes_flat)  # [T, B, N]
    in_spk = in_spk.squeeze(1).T
    hid_spk = hid_spk.squeeze(1).T
    out_spk = out_spk.squeeze(1).T

    # Plot raster
    plot_combined_raster(in_spk, hid_spk, out_spk, digit, f"./plots/smaller_raster_digit_{digit}.png")

    # Save cadence-style output flag .txt file for this digit
    save_output_spike_txt(out_spk, digit, save_dir="./plots", time_step_us=0.1)