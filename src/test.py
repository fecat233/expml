import torch

device = 'cpu'

if torch.cuda.is_available():
    device = 'cuda'

w = torch.randn(5,5, device=device)

print(w)
print("==========")
print(torch.cuda.get_device_capability(device=device))
print(torch.cuda.get_device_name(device=device))
print(torch.cuda.get_device_properties(device=device))