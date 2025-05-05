import matplotlib.pyplot as plt

# Input Data
results = {
    "4x4": {
        "epochs": [31.93, 33.40, 33.03, 33.12, 33.47],
        "quant": {2: 18.46, 3: 33.00, 4: 33.55, 5: 33.61},
        "t10": [42.19, 43.38, 43.20, 42.62, 42.84],
        "quant_t10": {2: 36.43, 3: 41.74, 4: 42.80, 5: 42.66},
        "t20": [53.00, 54.12, 53.76, 53.70, 53.92],
        "quant_t20": {2: 40.48, 3: 51.78, 4: 53.98, 5: 53.90}
    },
    "7x7": {
        "epochs": [66.36, 68.17, 69.15, 69.08, 70.02],
        "quant": {2: 20.01, 3: 66.40, 4: 69.75, 5: 69.89},
        "t10": [75.93, 78.31, 79.52, 79.60, 80.20],
        "quant_t10": {2: 37.17, 3: 75.77, 4: 79.87, 5: 80.03},
        "t20": [82.76, 84.39, 85.47, 86.13, 86.51],
        "quant_t20": {2: 12.63, 3: 82.80, 4: 85.76, 5: 86.37}
    },
    "14x14": {
        "epochs": [90.02, 91.36, 92.37, 93.37, 93.71],
        "quant": {2: 20.58, 3: 91.47, 4: 93.32, 5: 93.60},
        "t10": [91.39, 93.56, 94.48, 95.28, 95.50],
        "quant_t10": {2: 19.55, 3: 93.53, 4: 95.37, 5: 95.33},
        "t20": [92.90, 94.59, 95.59, 96.13, 96.33],
        "quant_t20": {2: 17.77, 3: 94.63, 4: 96.24, 5: 96.42}
    },
    "28x28": {
        "epochs": [94.32, 96.07, 96.51, 97.02, 97.10],
        "quant": {2: 11.77, 3: 96.60, 4: 96.77, 5: 97.10},
        "t10": [94.86, 96.11, 96.86, 97.00, 97.37],
        "quant_t10": {2: 13.53, 3: 97.32, 4: 97.33, 5: 97.18},
        "t20": [94.99, 96.56, 97.08, 97.29, 97.57],
        "quant_t20": {2: 11.65, 3: 96.89, 4: 97.60, 5: 97.55}
    }
}


img_sizes = ["4x4", "7x7", "14x14", "28x28"]
time_steps = [5, 10, 20]
quant_bits = [2, 3, 4, 5]
colors = ['red', 'green', 'blue']

# ---- PLOT 1: Accuracy vs Epochs ----
fig1, axs1 = plt.subplots(1, 4, figsize=(20, 5))
fig1.suptitle("Test Accuracy vs Epochs (by Image Size and Time Steps)", fontsize=16)

for i, size in enumerate(img_sizes):
    ax = axs1[i]
    for t, color in zip(time_steps, colors):
        key = 'epochs' if t == 5 else f't{t}'
        ax.plot(range(1, 6), results[size][key], marker='o', color=color, label=f'Time Steps: {t}')
    ax.set_title(size)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(True)
    if i == 0:
        ax.legend()

# ---- PLOT 2: Accuracy vs Time Steps (Quantization) ----
fig2, axs2 = plt.subplots(1, 4, figsize=(20, 5))
fig2.suptitle("Test Accuracy vs Time Steps (by Image Size and Quantization)", fontsize=16)

for i, size in enumerate(img_sizes):
    ax = axs2[i]
    for q in quant_bits:
        accs = [
            results[size]["quant"][q],
            results[size]["quant_t10"][q],
            results[size]["quant_t20"][q]
        ]
        ax.plot([5, 10, 20], accs, marker='o', label=f'{q}-bit')
    ax.set_title(size)
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(True)
    if i == 0:
        ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
