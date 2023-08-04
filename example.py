import torch
from nanox.model.encoder import NanoXGraphEncoder
from nanox.nanox import NanoXModel, NanoXModel

# Initialize the encoder
encoder = NanoXGraphEncoder()

# Initialize the model
model = NanoXModel(encoder)

# Define the batched data
batched_data = torch.rand(10, 512)  # Example data

# Forward pass through the model
output = model(batched_data)
