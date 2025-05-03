import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from spikingjelly.activation_based import neuron, encoding, functional, layer, monitor
#from spikingjelly.activation_based.learning import STDP
import os
import matplotlib.pyplot as plt
import csv
from pathlib import Path
import pandas as pd

# ==== CONFIGURATION ====
BATCH_SIZE = 64
EPOCHS = 5
# TIME_STEPS = 10
LEARNING_RATE = 1e-3
QUANTIZE_WEIGHTS = True
# QUANT_BITS = 3  # Choose from: 2, 3, 4, 5
TRAINING_MODE = "supervised"  # options: "supervised" -- Surrogate Gradient Descent, "unsupervised" -- STDP

# IMG_SIZE = [28, 14, 7, 4]
HIDDEN_NEURONS = 20
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
        self.fc1 = layer.Linear(IMG_SIZE * IMG_SIZE, HIDDEN_NEURONS)
        self.neuron1 = neuron.IFNode()
        self.fc2 = layer.Linear(HIDDEN_NEURONS, OUTPUT_NEURONS)
        self.neuron2 = neuron.IFNode()

        # Add monitors
        self.mon1 = monitor.OutputMonitor(self.neuron1)
        self.mon2 = monitor.OutputMonitor(self.neuron2)

    def forward(self, x_spike):
        self.mon1 = monitor.OutputMonitor(self.neuron1)
        self.mon2 = monitor.OutputMonitor(self.neuron2)

        for t in range(x_spike.shape[0]):
            x = x_spike[t].view(x_spike.size(1), -1)
            x = self.fc1(x)
            x = self.neuron1(x)
            x = self.fc2(x)
            x = self.neuron2(x)
        return x



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
        spike_input = encoder(images).float()
        outputs = model(spike_input)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        functional.reset_net(model)
    return 100 * correct / total

# ==== STDP Training ====
# def train_with_stdp(model, train_loader, encoder):
#     # Only for one epoch (unsupervised loop)
#     model.train()
    
#     # Wrap layers with STDP
#     stdp_fc1 = STDP(model.fc1, model.neuron1, f_pre=1.0, f_post=-1.0)
#     stdp_fc2 = STDP(model.fc2, model.neuron2, f_pre=1.0, f_post=-1.0)

#     for images, _ in train_loader:  # no labels used
#         images = images.to(DEVICE)
#         spike_input = encoder(images.float()).float()
        
#         for t in range(TIME_STEPS):
#             x_t = spike_input[t].view(spike_input.size(1), -1)
#             out1 = model.fc1(x_t)
#             out1 = model.neuron1(out1)
#             out2 = model.fc2(out1)
#             out2 = model.neuron2(out2)

#             stdp_fc1.step(x_t, out1)
#             stdp_fc2.step(out1, out2)

#         functional.reset_net(model)

# ==== Raster Plots ====
def plot_spike_raster_manual(spike_tensor, save_path, title):
    """
    Custom raster plotter for [time_steps, neurons] spike tensor.
    Saves directly to PNG (no display).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    spike_array = spike_tensor.cpu().numpy()
    t_idxs, n_idxs = spike_array.nonzero()  # Find all (time, neuron) spike pairs

    plt.figure(figsize=(8, 4))
    plt.scatter(t_idxs, n_idxs, s=2, c='black')
    plt.xlabel("Time step")
    plt.ylabel("Neuron index")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ==== Export Raster Data ====
def export_spike_raster_to_csv(spike_tensor, csv_path):
    """
    Exports spike raster (time_steps x neurons) to a CSV file.
    Each row contains the time step and neuron index for a spike.
    """
    spike_array = spike_tensor.cpu().numpy()
    rows = []
    for t in range(spike_array.shape[0]):
        for n in range(spike_array.shape[1]):
            if spike_array[t, n]:  # Spike occurred
                rows.append({"time_step": t, "neuron_index": n})

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"Saved spike raster to {csv_path}")


# ==== MAIN FUNCTION ====
def main():
    image_sizes = [4, 7, 14, 28]
    quant_bits_list = [2, 3, 4, 5]
    time_step_list = [5, 10, 20]

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
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

            # Model + Encoder
            encoder = PoissonEncoder(time_steps=TIME_STEPS)
            model = SNN().to(DEVICE)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

            # Training
            for epoch in range(EPOCHS):
                print(f"Epoch {epoch+1}/{EPOCHS}")
                
                if TRAINING_MODE == "supervised":
                    model.train()
                    running_loss = 0.0
                    for images, labels in train_loader:
                        images = images.to(DEVICE)
                        labels = labels.to(DEVICE)
                        spike_input = encoder(images.float()).float()
                        outputs = model(spike_input)
                        loss = criterion(outputs, labels)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        functional.reset_net(model)
                        running_loss += loss.item()
                    print(f"  Loss: {running_loss:.4f}")

                elif TRAINING_MODE == "unsupervised":
                    #train_with_stdp(model, train_loader, encoder)
                    print("  STDP update complete.")

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
                    with open("snn_hardware_results.csv", mode="a", newline="") as file:
                        writer = csv.DictWriter(file, fieldnames=["img_size", "time_steps", "quant_bits", "test_accuracy"])
                        writer.writerow({
                            "img_size": img_size,
                            "time_steps": time_steps,
                            "quant_bits": bits,
                            "test_accuracy": test_acc_q
                        })

            print(f"=== Done: IMG {img_size} | TIME_STEPS {time_steps} ===\n")

            # === EXPORT SPIKE RASTERS AND CSV FOR MULTIPLE DIGITS ===
            model.eval()
            example_digits = [0, 6, 8]

            for digit in example_digits:
                sample_idx = next(i for i, (_, y) in enumerate(test_dataset) if y == digit)
                image, label = test_dataset[sample_idx]
                image = image.unsqueeze(0).to(DEVICE)
                spike_input = encoder(image).float()

                # Save input spike raster
                input_spike_record = spike_input.squeeze(1).view(TIME_STEPS, -1)  # Flatten input to [T, N]
                plot_spike_raster_manual(
                    title=f"Input layer spikes - Digit {digit}",
                    spike_record=input_spike_record,
                    save_path=f"plots/input_spikes_digit_hardware_{IMG_SIZE}_{digit}.png"
                )

                _ = model(spike_input)

                # Save hidden and output spike rasters
                plot_spike_raster_manual(
                    title=f"Hidden layer spikes - Digit {digit}",
                    spike_record=model.mon1.records[0],
                    save_path=f"plots/hidden_spikes_digit_hardware_{IMG_SIZE}_{digit}.png"
                )
                plot_spike_raster_manual(
                    title=f"Output layer spikes - Digit {digit}",
                    spike_record=model.mon2.records[0],
                    save_path=f"plots/output_spikes_digit_hardware_{IMG_SIZE}_{digit}.png"
                )

                # Save output spike raster to CSV
                export_spike_raster_to_csv(
                    model.mon2.records[0],
                    csv_path=f"plots/output_spike_raster_digit_hardware_{IMG_SIZE}_{digit}.csv"
                )

if __name__ == '__main__':
    main()
