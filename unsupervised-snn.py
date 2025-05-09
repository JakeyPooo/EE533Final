import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from spikingjelly.activation_based import neuron, layer, learning, functional
from collections import defaultdict
import csv

# ==== CONFIG ====
HIDDEN_NEURONS = 512
OUTPUT_NEURONS = 10
EPOCHS = 5
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
QUANT_BITS = [2, 3, 4, 5]

# ==== Encoder ====
class PoissonEncoder(nn.Module):
    def __init__(self, time_steps):
        super().__init__()
        self.time_steps = time_steps

    def forward(self, x):
        return (torch.rand((self.time_steps,) + x.shape, device=x.device) < x).float()

# ==== STDP Network ====
class STDPNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = layer.Linear(in_dim, hidden_dim, bias=False)
        self.n1 = neuron.IFNode()
        self.fc2 = layer.Linear(hidden_dim, out_dim, bias=False)
        self.n2 = neuron.IFNode()

        self.stdp1 = learning.STDPLearner('s', self.fc1, self.n1, 2.0, 2.0,
                                          f_pre=lambda x: torch.clamp(x, -1., 1.),
                                          f_post=lambda x: torch.clamp(x, -1., 1.))
        self.stdp2 = learning.STDPLearner('s', self.fc2, self.n2, 2.0, 2.0,
                                          f_pre=lambda x: torch.clamp(x, -1., 1.),
                                          f_post=lambda x: torch.clamp(x, -1., 1.))

    def forward(self, x_seq, stdp_train=False):
        if x_seq.dim() == 5:
            x_seq = x_seq.squeeze(2)  # [T, B, 1, H, W] → [T, B, H, W]
        spike_record = torch.zeros(x_seq.size(1), OUTPUT_NEURONS, device=x_seq.device)
        for t in range(x_seq.shape[0]):
            x = x_seq[t].view(x_seq.size(1), -1)
            h = self.fc1(x)
            h = self.n1(h)
            o = self.fc2(h)
            s = self.n2(o)
            spike_record += s
            if stdp_train:
                self.stdp1.step(on_grad=True)
                self.stdp2.step(on_grad=True)
                self.fc1.weight.data.clamp_(-1., 1.)
                self.fc2.weight.data.clamp_(-1., 1.)
        return spike_record

# ==== Accuracy Functions ====
def assign_neuron_labels(model, loader, encoder):
    label_map = defaultdict(lambda: torch.zeros(OUTPUT_NEURONS, device=DEVICE))
    model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            spikes = encoder(imgs)
            functional.reset_net(model)
            out = model(spikes, stdp_train=False)
            for i in range(out.size(0)):
                nid = torch.argmax(out[i]).item()
                label_map[nid][labels[i]] += 1
    return {nid: torch.argmax(c).item() for nid, c in label_map.items()}

@torch.no_grad()
def evaluate_accuracy(model, loader, neuron_labels, encoder):
    correct = 0
    total = 0
    model.eval()
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        spikes = encoder(imgs)
        functional.reset_net(model)
        out = model(spikes, stdp_train=False)
        preds = torch.argmax(out, dim=1)
        for i in range(len(preds)):
            pred_label = neuron_labels.get(preds[i].item(), -1)
            if pred_label == labels[i].item():
                correct += 1
            total += 1
    return 100 * correct / total

def quantize_model_weights(model, num_bits):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name:
                w_min = param.min()
                w_max = param.max()
                delta = (w_max - w_min) / (2 ** num_bits - 1)
                param.copy_(torch.round((param - w_min) / delta) * delta + w_min)

# ==== MAIN LOOP ====
def main():
    sizes = [7] #[4, 7, 14, 28]
    time_steps_list = [40]#[5, 10, 20]

    for size in sizes:
        for tsteps in time_steps_list:
            print(f"=== SIZE: {size}x{size}, TIME_STEPS: {tsteps} ===")

            transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor()
            ])
            encoder = PoissonEncoder(tsteps)
            train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
            test_set = datasets.MNIST('./data', train=False, download=True, transform=transform)

            train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

            in_dim = size * size
            model = STDPNet(in_dim, HIDDEN_NEURONS, OUTPUT_NEURONS).to(DEVICE)

            for epoch in range(EPOCHS):
                model.train()
                for imgs, _ in train_loader:
                    imgs = imgs.to(DEVICE)
                    spikes = encoder(imgs)
                    functional.reset_net(model)
                    model(spikes, stdp_train=True)
                neuron_labels = assign_neuron_labels(model, train_loader, encoder)
                acc = evaluate_accuracy(model, test_loader, neuron_labels, encoder)
                print(f"  Epoch {epoch+1} Accuracy: {acc:.2f}%")
                torch.cuda.empty_cache()

            for bits in QUANT_BITS:
                model_q = STDPNet(in_dim, HIDDEN_NEURONS, OUTPUT_NEURONS).to(DEVICE)
                model_q.load_state_dict(model.state_dict())
                quantize_model_weights(model_q, bits)
                acc_q = evaluate_accuracy(model_q, test_loader, neuron_labels, encoder)
                print(f"  Quantized @ {bits}-bit → Accuracy: {acc_q:.2f}%")

                with open("stdp_snn_results.csv", mode="a", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=["img_size", "time_steps", "quant_bits", "accuracy"])
                    writer.writeheader()
                    writer.writerow({
                        "img_size": size,
                        "time_steps": tsteps,
                        "quant_bits": bits,
                        "accuracy": acc_q
                    })

if __name__ == '__main__':
    main()