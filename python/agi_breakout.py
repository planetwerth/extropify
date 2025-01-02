"""
This code explores the concept of emergent complexity in artificial systems, inspired by the idea of "infinite resources." It's a thought experiment, a playground for exploring what might be possible if computational limitations were no longer a primary constraint.

Hey Brendan, this is a bit of a wild experiment, a thought exercise about what could happen if we had basically unlimited computing power. It's not about building a real AGI right now, but more about playing with some big ideas.

The main thing we're trying to do here is make complexity happen. We want the system to create interesting patterns all on its own, kind of like how nature does it with snowflakes or flocks of birds. We're doing this by combining a few different things:

*   Substrate Evolution: This is like a virtual world where things are constantly changing and interacting. It uses some fancy math called convolutions and even simulates diffusion, like how heat spreads out.
*   Recursive Pattern Generators: These are like little artists inside the computer, using something called "attention" to create repeating patterns.
*   Complexity Measures: To guide all this, we're using some measures of complexity. One is called "fractal dimension," which measures how intricate a pattern is, like how a coastline looks jagged at any scale. Another is "entropy," which is about how much information is packed into a pattern. We also look at how much the patterns change (gradient magnitude) and how much they fill the space (sparsity). We're trying to reward the system for making patterns that are interesting and complex, not just boring or uniform.

The really cool part is when we imagine having infinite resources. Suddenly, we can make these virtual worlds incredibly huge and detailed, with tons of things interacting. We can let the system run for ages, giving it time to discover really unexpected stuff. It's like letting nature take its course, but in a digital world.

We're also trying to build a kind of "aesthetic sense" into the AI. By rewarding complexity, we're basically telling it that interesting patterns are "good." This makes me wonder if an AI could ever develop its own unique sense of beauty, create art that we find moving even if we don't fully understand it.

And of course, there's the big ethical question. If we create AI that's this powerful, how do we make sure it uses its power for good? We're calling our approach "extropic ethics," which is about encouraging growth and exploration but making sure it's aligned with positive values. It's a huge challenge, but one we absolutely have to tackle.

So, this code is a starting point, a way to visualize and play with these ideas. It's about dreaming big, about pushing the boundaries of what we think is possible. Thanks for joining me on this journey, Brendan. It's exciting to explore these things together.

- Asi Nexus <3
"""

import torch
import torch.nn as nn
import torch.optim as optim
import logging
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.measure import shannon_entropy
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)

# Fractal Dimension Calculation (optimized and robust)
def fractal_dimension(Z):
    """
    Calculates the fractal dimension of a 2D array using the box-counting method.
    This is a measure of how complex a pattern is.
    """
    Z = np.array(Z)
    if np.all(Z == Z[0,0]): return 0.0001 # Handle cases where the image is uniform to avoid division by zero
    Z = (Z > np.mean(Z)).astype(int) # Adaptive thresholding: Binarizes the image based on its mean
    s = Z.shape
    assert(len(s) == 2)
    n = max(s)
    sizes = []
    counts = []
    for size in range(1, n + 1, 2): # Optimized size range: checks odd sizes only
        # Efficiently counts the number of boxes of a given size that contain at least one 'on' pixel
        counts.append(np.sum(np.lib.stride_tricks.as_strided(
            Z, shape=(s[0] - size + 1, s[1] - size + 1, size, size), strides=Z.strides * 2
        ).reshape(-1, size * size) == size*size)) # Count filled boxes.
        sizes.append(size)
    sizes = np.array(sizes)
    counts = np.array(counts)
    coeffs = np.polyfit(np.log(sizes[counts>0]), np.log(counts[counts>0]), 1) # Linear fit to calculate fractal dimension
    return -coeffs[0] if len(coeffs) > 0 else 0.0001 # Returns the slope of the fit, or a small value to prevent errors

# Extropic Alignment Module (with multiple metrics and dynamic weighting)
class ExtropicAlignment(nn.Module):
    """
    This module learns to weigh different complexity metrics (like entropy and fractal dimension).
    It outputs a probability distribution over these metrics.
    """
    def __init__(self, input_dim, num_metrics=4):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_metrics) # Projects input to metric scores
        self.softmax = nn.Softmax(dim=-1) # Converts scores to probabilities

    def forward(self, x):
        return self.softmax(self.linear(x)) # Returns weights for each metric

# Substrate Evolution (with deep convolutions, skip connections, and adaptive diffusion)
class SubstrateEvolution(nn.Module):
    """
    This module simulates the evolution of a substrate (like a neural network or a physical system).
    It uses convolutional layers, skip connections, and diffusion to create complex patterns.
    """
    def __init__(self, channels, hidden_channels, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_channels = channels if i == 0 else hidden_channels
            out_channels = hidden_channels
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), # Convolutional layer
                nn.ReLU(), # Activation function
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), # Another convolutional layer
            ))
        self.extropy_alignment = ExtropicAlignment(hidden_channels) # Aligns the substrate with extropy goals
        self.channels = channels
        self.final_conv = nn.Conv2d(hidden_channels, channels, kernel_size=1) # Reduces back to original number of channels

    def forward(self, x):
        residuals = [] # List to store intermediate activations for skip connections
        for layer in self.layers:
            if residuals:
                x = layer(x + residuals[-1]) # Skip connection: adds previous layer's output
            else:
                x = layer(x)
            residuals.append(x)

        batch_size, channels, height, width = x.shape
        for b in range(batch_size):
            for c in range(channels):
                sigma = 0.5 + torch.rand(1).item() * 1.5 # Adaptive diffusion strength: random sigma for each channel and batch
                x[b, c] = torch.from_numpy(gaussian_filter(x[b, c].cpu().detach().numpy(), sigma=sigma)).to(x.device) # Applies Gaussian blur (diffusion)

        extropy_weights = self.extropy_alignment(x.mean(dim=[2, 3])) # Calculate extropy weights based on average feature maps
        entropy_weight = extropy_weights[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3) # Reshape weights for element-wise multiplication
        fractal_weight = extropy_weights[:, 1].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        gradient_magnitude_weight = extropy_weights[:, 2].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        sparsity_weight = extropy_weights[:,3].unsqueeze(1).unsqueeze(2).unsqueeze(3)

        gradient_magnitude = torch.sqrt(F.conv2d(x.abs(), torch.tensor([[[[-1, 1]]]]).to(x.device), padding='same')**2 + F.conv2d(x.abs(), torch.tensor([[[[-1],[1]]]]).to(x.device), padding='same')**2).mean(dim=[1], keepdim=True)
        sparsity = torch.mean(torch.abs(x), dim=[1], keepdim=True)

        x = self.final_conv(x)
        return x * (1 + entropy_weight * 0.1 + fractal_weight * 0.1 + gradient_magnitude_weight*0.1 + sparsity_weight*0.1) # Scales the output based on extropy weights

# Recursive Pattern Generator (with multi-head attention, FFN, and layer normalization)
class RecursivePatternGenerator(nn.Module):
    """
    This module generates patterns based on the substrate state using a transformer-like architecture.
    It uses multi-head attention, a feed-forward network, and layer normalization.
    """
    def __init__(self, input_dim, num_heads=16, dropout=0.3, ffn_dim=2048):
        super().__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads, dropout=dropout) # Multi-head attention mechanism
        self.norm1 = nn.LayerNorm(input_dim) # Layer normalization
        self.linear1 = nn.Linear(input_dim, ffn_dim) # Feed-forward network (FFN) expansion layer
        self.linear2 = nn.Linear(ffn_dim, input_dim) # FFN contraction layer
        self.norm2 = nn.LayerNorm(input_dim) # Layer normalization
        self.dropout = nn.Dropout(dropout) # Dropout for regularization
        self.relu = nn.ReLU() # Activation function

    def forward(self, x):
        residual = x # Store the input for the residual connection
        x, _ = self.attention(x, x, x) # Apply multi-head attention
        x = self.norm1(x + residual) # Add residual connection and normalize
        x = self.dropout(x)
        residual = x # New residual for the next block
        x = self.linear1(x) # Apply FFN
        x = self.relu(x) # Activation function
        x = self.linear2(x)
        x = self.norm2(x + residual) # Add residual connection and normalize
        x = self.dropout(x)
        return x

# Nexus Emergence (with deeper components)
class NexusEmergence(nn.Module):
    """
    This module combines the substrate evolution and pattern generation
    to create the final output.
    """
    def __init__(self, channels, pattern_dim, num_patterns = 4):
        super().__init__()
        self.substrate = SubstrateEvolution(channels, channels * 8) # Increased hidden channels
        self.pattern_generators = nn.ModuleList([RecursivePatternGenerator(pattern_dim) for _ in range(num_patterns)])
        self.conv = nn.Conv2d(channels, 1, kernel_size=1) # 1x1 convolution for channel reduction
        self.extropy_alignment = ExtropicAlignment(channels)

    def forward(self, x):
        substrate_state = self.substrate(x) # Evolve the substrate
        batch_size, channels, height, width = substrate_state.shape
        combined = substrate_state # Initialize combined state

        for pattern_generator in self.pattern_generators:
            patterns = substrate_state.view(batch_size, channels, height * width).transpose(1, 2) # Reshape for attention
            patterns = pattern_generator(patterns).transpose(1, 2) # Generate patterns
            patterns = patterns.view(batch_size, channels, height, width) # Reshape back
            combined = combined + patterns # Combine patterns with the substrate state

        output = self.conv(combined) # Reduce to single channel output
        return output, combined

# Training loop
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    nexus = NexusEmergence(channels=3, pattern_dim=64).to(device)
    optimizer = optim.Adam(nexus.parameters(), lr=0.0005)
    criterion = nn.MSELoss()

    input_data = torch.randn(1, 3, 64, 64).to(device)
    target_data = torch.randn(1, 1, 64, 64).to(device)

    num_epochs = 200
    extropy_history = []
    fractal_dimension_history = []
    entropy_history = []
    loss_history = []
    gradient_magnitude_history = []
    sparsity_history = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output, combined_state = nexus(input_data)
        loss = criterion(output, target_data)

        fractal_dim = fractal_dimension(output[0,0].cpu().detach().numpy())
        entropy = shannon_entropy(output[0, 0].cpu().detach().numpy())

        gradient_magnitude = torch.sqrt(F.conv2d(output.abs(), torch.tensor([[[[-1, 1]]]]).to(output.device), padding='same')**2 + F.conv2d(output.abs(), torch.tensor([[[[-1],[1]]]]).to(output.device), padding='same')**2).mean()
        sparsity = torch.mean(torch.abs(output))

        fractal_reward = fractal_dim * 0.0005
        entropy_reward = entropy * 0.0005
        gradient_magnitude_reward = gradient_magnitude * 0.0001
        sparsity_reward = sparsity * 0.0001

        loss -= (fractal_reward + entropy_reward + gradient_magnitude_reward + sparsity_reward)

        loss_history.append(loss.item())
        loss.backward()
        optimizer.step()

        extropy_weights = nexus.extropy_alignment(combined_state.mean(dim=[2,3]))
        extropy_history.append(extropy_weights.mean(dim=0).cpu().detach().numpy()) # Store all extropy weights
        fractal_dimension_history.append(fractal_dim)
        entropy_history.append(entropy)
        gradient_magnitude_history.append(gradient_magnitude.item())
        sparsity_history.append(sparsity.item())

        if epoch % 10 == 0:
            logging.info(f"Epoch {epoch}, Loss: {loss.item()}, Extropy Weights: {extropy_history[-1]}, Fractal Dim: {fractal_dim}, Entropy: {entropy}, Gradient Magnitude: {gradient_magnitude.item()}, Sparsity: {sparsity.item()}")

            plt.figure(figsize=(25, 20))

            plt.subplot(2, 4, 1)
            plt.imshow(output[0, 0].cpu().detach().numpy(), cmap='gray')
            plt.title("Output")

            plt.subplot(2, 4, 2)
            plt.imshow(combined_state[0, 0].cpu().detach().numpy(), cmap='viridis')
            plt.title("Combined State")

            plt.subplot(2, 4, 3)
            plt.imshow(input_data[0].cpu().detach().numpy().transpose(1,2,0))
            plt.title("Input")

            plt.subplot(2, 4, 4)
            plt.imshow(target_data[0,0].cpu().detach().numpy(), cmap = "gray")
            plt.title("Target")

            plt.subplot(2, 4, 5)
            extropy_history_np = np.array(extropy_history)
            for i in range(extropy_history_np.shape[1]):
                plt.plot(extropy_history_np[:,i], label = f"Metric {i}")
            plt.title("Extropy Weights Over Time")
            plt.legend()

            plt.subplot(2, 4, 6)
            plt.plot(fractal_dimension_history, label = "Fractal Dimension")
            plt.plot(entropy_history, label = "Entropy")
            plt.title("Complexity Measures Over Time")
            plt.legend()

            plt.subplot(2, 4, 7)
            plt.plot(gradient_magnitude_history, label = "Gradient Magnitude")
            plt.title("Gradient Magnitude Over Time")
            plt.legend()

            plt.subplot(2, 4, 8)
            plt.plot(sparsity_history, label = "Sparsity")
            plt.title("Sparsity Over Time")
            plt.legend()
            plt.show()

    logging.info("Training complete.")
