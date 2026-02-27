import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import utils.plot_style

def get_positional_encoding(max_seq_len, d_model):
    pe = np.zeros((max_seq_len, d_model))
    for pos in range(max_seq_len):
        for i in range(0, d_model, 2):
            div_term = np.exp(i * -np.log(10000.0) / d_model)
            pe[pos, i] = np.sin(pos * div_term)
            if i + 1 < d_model:
                pe[pos, i + 1] = np.cos(pos * div_term)
    return pe

d_model = 128
max_seq_len = 50
pe = get_positional_encoding(max_seq_len, d_model)

plt.figure(figsize=(10, 6))
plt.imshow(pe, cmap='viridis', aspect='auto')
plt.title("Sinusoidal Positional Encoding")
plt.xlabel("Encoding Dimension (d_model)")
plt.ylabel("Position in Sequence")
plt.colorbar(label="Value")
plt.tight_layout()
plt.savefig('chapter_03/images/positional_encoding.png')
print("Positional encoding plot saved to chapter_03/images/positional_encoding.png")
