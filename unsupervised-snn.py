import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer
from collections import defaultdict
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 5
HIDDEN_NEURONS = 100
STDP_LR = 0.005
QUANTIZE_WEIGHTS = True

# ==== Quantization ====
def quantize_weights(model, num_bits):
    with torch.no_grad():
        w = model.fc.weight
        w_min = w.min()
        w_max = w.max()
        delta = (w_max - w_min) / (2 ** num_bits - 1)
        w_q = torch.round((w - w_min) / delta) * delta + w_min
        model.fc.weight.data.copy_(w_q)

# ==== Poisson Encoder ====
class PoissonEncoder(torch.nn.Module):
    def __init__(self, time_steps):
        super().__init__()
        self.time_steps = time_steps

    def forward(self, x):
        return (torch.rand((self.time_steps,) + x.shape, device=x.device) < x).float()

# ==== STDP Network ====
class STDPNetLabelAssign(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc = layer.Linear(input_dim, hidden_dim, bias=False)
        self.neuron = neuron.IFNode(surrogate_function=surrogate.Sigmoid())
        self.stdp_lr = STDP_LR

    def forward(self, x_seq):
        batch_size = x_seq.size(1)
        device = x_seq.device
        spike_trace = torch.zeros((batch_size, self.fc.out_features), device=device)

        for t in range(x_seq.shape[0]):
            x = x_seq[t].view(batch_size, -1).float()
            mem = self.fc(x)
            s_out = self.neuron(mem)
            pre = x.unsqueeze(2)
            post = s_out.unsqueeze(1)
            dw = self.stdp_lr * torch.bmm(pre, post)
            self.fc.weight.data += dw.mean(0).T
            spike_trace += s_out

        return spike_trace

# ==== Neuron-to-Label Assignment ====
def assign_neuron_labels(model, data_loader, encoder):
    label_counts = defaultdict(lambda: torch.zeros(10, device=DEVICE))

    model.eval()
    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            spikes = encoder(imgs).squeeze(2).float()
            functional.reset_net(model)
            out_spikes = model(spikes)

            for b in range(out_spikes.size(0)):
                label = labels[b].item()
                neuron_id = torch.argmax(out_spikes[b]).item()
                label_counts[neuron_id][label] += 1

    neuron_labels = {}
    for neuron_id, counts in label_counts.items():
        neuron_labels[neuron_id] = torch.argmax(counts).item()
    return neuron_labels

# ==== Accuracy Testing ====
def test_accuracy(model, data_loader, encoder, neuron_labels):
    correct = 0
    total = 0
    model.eval()

    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            spikes = encoder(imgs).squeeze(2).float()
            functional.reset_net(model)
            out_spikes = model(spikes)

            preds = torch.argmax(out_spikes, dim=1)
            mapped_preds = torch.tensor([neuron_labels.get(pid.item(), -1) for pid in preds]).to(DEVICE)

            correct += (mapped_preds == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total

# ==== Main Experiment Loop ====
def main():
    image_sizes = [4, 7, 14, 28]
    time_step_list = [5, 10, 20]
    quant_bit_list = [2, 3, 4, 5]

    for img_size in image_sizes:

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        input_dim = img_size * img_size

        for time_steps in time_step_list:
            print(f"\n======= IMG SIZE: {img_size}x{img_size}, TIME_STEPS: {time_steps} =======")
            encoder = PoissonEncoder(time_steps)

            model = STDPNetLabelAssign(input_dim, HIDDEN_NEURONS).to(DEVICE)
            for epoch in range(EPOCHS):
                model.train()
                for imgs, _ in train_loader:
                    imgs = imgs.to(DEVICE)
                    spikes = encoder(imgs).squeeze(2).float()
                    functional.reset_net(model)
                    model(spikes)
                print(f"  Epoch {epoch+1}/{EPOCHS} done")

            neuron_labels = assign_neuron_labels(model, train_loader, encoder)
            acc = test_accuracy(model, test_loader, encoder, neuron_labels)
            print(f"  Unquantized Test Accuracy: {acc:.2f}%")

            for bits in quant_bit_list:
                model_q = STDPNetLabelAssign(input_dim, HIDDEN_NEURONS).to(DEVICE)
                model_q.load_state_dict(model.state_dict())
                quantize_weights(model_q, bits)
                acc_q = test_accuracy(model_q, test_loader, encoder, neuron_labels)
                print(f"  Quantized @ {bits}-bit → Test Accuracy: {acc_q:.2f}%")

if __name__ == '__main__':
    main()
