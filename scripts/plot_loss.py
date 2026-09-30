import numpy as np
import matplotlib.pyplot as plt
from config import CONFIG

base = f'dt{CONFIG["dt"]:.2f}_w{CONFIG["washout_length"]}'
loss_curves = np.load(f'{base}_loss.npy', allow_pickle=True).item()

plt.figure(figsize=(10, 6))

for model_key, mem_dict in loss_curves.items():
    for mem_size, curve in mem_dict.items():
        label = f'{model_key}_M{mem_size}'
        plt.plot(curve, label=label, linewidth=1.5)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'All MLP Loss Curves (dt={CONFIG["dt"]:.2f}, washout={CONFIG["washout_length"]})')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()

base = f'dt{CONFIG["dt"]:.2f}_w{CONFIG["washout_length"]}'
loss_curves = np.load(f'{base}_loss.npy', allow_pickle=True).item()

data = []
labels = []

for model_key, mem_dict in loss_curves.items():
    for mem_size, curve in mem_dict.items():
        data.append(curve)               # 1D array of per-epoch losses
        labels.append(f'{model_key}_M{mem_size}')

plt.figure(figsize=(10, 6))
plt.boxplot(data, labels=labels, showfliers=True)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Loss')
plt.title(f'Loss Distribution per Model/Memory (dt={CONFIG["dt"]:.2f}, washout={CONFIG["washout_length"]})')
plt.tight_layout()
plt.show()
