import torch

loaded_tensor = torch.load('./video335.mp4.pt')

# Print the loaded PyTorch tensor
print(type(loaded_tensor))

print(loaded_tensor.shape)