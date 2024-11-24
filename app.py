import streamlit as st
import torch
import torch.nn as nn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# Define the Generator class
class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self._block(z_dim, features_g * 16, 4, 1, 0),  # 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # 32x32
            nn.ConvTranspose2d(features_g * 2, channels_img, 4, 2, 1),  # 64x64
            nn.Tanh()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.gen(x)

# Load the generator model with architecture and weights
@st.cache_resource
def load_generator_model():
    generator_path = "models/generator_full.pth"  # Path to the .pth file in the repo

    # Load the full generator model (architecture + weights)
    generator = torch.load(generator_path, map_location=torch.device("cpu"))
    generator.eval()
    return generator

# Generate images using the generator
def generate_images(generator, latent_dim, num_images):
    noise = torch.randn(num_images, latent_dim, 1, 1)
    with torch.no_grad():
        images = generator(noise).cpu()
    return images

# Streamlit App
st.title("DCGAN Image Generator")

# Load the generator
generator = load_generator_model()
latent_dim = 100  # Latent vector size

# User input for the number of images to generate
num_images = st.slider("Number of images to generate", 1, 16, 4)

if st.button("Generate Images"):
    images = generate_images(generator, latent_dim, num_images)
    images = (images + 1) / 2  # Rescale images to [0, 1]

    # Display images
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis("off")
    ax.imshow(make_grid(images, nrow=4).permute(1, 2, 0))
    st.pyplot(fig)
