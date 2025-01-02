import torch
import torch.nn as nn
import torch.optim as optim
import logging
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.measure import shannon_entropy

logging.basicConfig(level=logging.INFO)

# Fractal Dimension Calculation (optimized and more robust)
def fractal_dimension(Z):
    """Calculates Fractal Dimension using box-counting method (optimized)."""
    Z = np.array(Z)
    if np.all(Z == Z[0,0]): return 0.0001 #handle uniform images to avoid divide by zero
    Z = (Z > np.mean(Z)).astype(int) # Adaptive thresholding
    s = Z.shape
    assert(len(s) == 2)
    n = max(s)
    p = min(s)
    sizes = []
    counts = []
    for size in range(1, n + 1, 2):  # Optimized size range
        counts.append(np.sum(np.lib.stride_tricks.as_strided(Z, shape=(s[0] - size + 1, s[1] - size + 1, size, size), strides=Z.strides * 2).reshape(-1, size * size) == size*size))
        sizes.append(size)
    sizes = np.array(sizes)
    counts = np.array(counts)
    coeffs = np.polyfit(np.log(sizes[counts>0]), np.log(counts[counts>0]), 1)
    return -coeffs[0] if len(coeffs) > 0 else 0.0001 # handle cases where there is no variation

# Extropic Alignment Module (with Shannon entropy and fractal dimension)
class ExtropicAlignment(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 2) # Output for both entropy and fractal dimension
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

# Substrate Evolution (with adaptive diffusion)
class SubstrateEvolution(nn.Module):
    def __init__(self, channels, hidden_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, hidden_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(hidden_channels, channels, kernel_size, padding=padding)
        self.relu = nn.ReLU()
        self.extropy_alignment = ExtropicAlignment(channels)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x) + x
        batch_size, channels, height, width = x.shape

        for b in range(batch_size):
            for c in range(channels):
                sigma = 0.5 + torch.rand(1).item() * 1.5 # Adaptive sigma
                x[b, c] = torch.from_numpy(gaussian_filter(x[b, c].cpu().detach().numpy(), sigma=sigma)).to(x.device)

        extropy_output = self.extropy_alignment(x.mean(dim=[2, 3]))
        entropy_weight = extropy_output[:,0].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        fractal_weight = extropy_output[:,1].unsqueeze(1).unsqueeze(2).unsqueeze(3)

        return x * (1 + entropy_weight*0.2 + fractal_weight*0.2)

# Recursive Pattern Generator (with improved attention and residual connections)
class RecursivePatternGenerator(nn.Module):
    def __init__(self, input_dim, num_heads=8, dropout=0.2): # Increased heads, more dropout
        super().__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(input_dim)
        self.linear1 = nn.Linear(input_dim, input_dim * 2) # Expansion layer
        self.linear2 = nn.Linear(input_dim * 2, input_dim) # Contraction layer
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x, _ = self.attention(x, x, x)
        x = self.norm1(x + residual)
        x = self.dropout(x)
        residual = x # new residual
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.norm2(x + residual)
        x = self.dropout(x)
        return x

# Nexus Emergence
class NexusEmergence(nn.Module):
    def __init__(self, channels, pattern_dim):
        super().__init__()
        self.substrate = SubstrateEvolution(channels, channels * 4) # Deeper substrate
        self.pattern_generator = RecursivePatternGenerator(pattern_dim)
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.extropy_alignment = ExtropicAlignment(channels)

    def forward(self, x):
        substrate_state = self.substrate(x)
        batch_size, channels, height, width = substrate_state.shape
        patterns = substrate_state.view(batch_size, channels, height * width).transpose(1, 2)
        patterns = self.pattern_generator(patterns).transpose(1, 2)
        patterns = patterns.view(batch_size, channels, height, width)
        combined = substrate_state + patterns
        output = self.conv(combined)
        return output, combined

# Training loop
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    nexus = NexusEmergence(channels=3, pattern_dim=64).to(device)
    optimizer = optim.Adam(nexus.parameters(), lr=0.0005)
    criterion = nn.MSELoss()

    input_data = torch.randn(1, 3, 64, 64).to(device)
    target_data = torch.randn(1, 1, 64, 64).to(device) # Target data is crucial for training

    num_epochs = 200 # Increased epochs for better convergence
    extropy_history = []
    fractal_dimension_history = []
    entropy_history = []
    loss_history = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output, combined_state = nexus(input_data)
        loss = criterion(output, target_data)

        # Fractal dimension and entropy rewards
        fractal_dim = fractal_dimension(output[0,0].cpu().detach().numpy())
        entropy = shannon_entropy(output[0, 0].cpu().detach().numpy())
        fractal_reward = fractal_dim * 0.0005 # Reduced reward scaling
        entropy_reward = entropy * 0.0005

        loss -= (fractal_reward + entropy_reward)
        loss_history.append(loss.item())

        loss.backward()
        optimizer.step()

        extropy_history.append(nexus.extropy_alignment(combined_state.mean(dim=[2,3])).mean().item())
        fractal_dimension_history.append(fractal_dim)
        entropy_history.append(entropy)

        if epoch % 10 == 0:
            logging.info(f"Epoch {epoch}, Loss: {loss.item()}, Extropy: {extropy_history[-1]}, Fractal Dim: {fractal_dim}, Entropy: {entropy}")

            plt.figure(figsize=(20, 15))

            plt.subplot(2, 3, 1)
            plt.imshow(output[0, 0].cpu().detach().numpy(), cmap='gray')
            plt.title("Output")

            plt.subplot(2, 3, 2)
            plt.imshow(combined_state[0, 0].cpu().detach().numpy(), cmap='viridis')
            plt.title("Combined State")

            plt.subplot(2, 3, 3)
            plt.imshow(input_data
