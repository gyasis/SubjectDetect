# %%
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA device is available")
else:
    device = torch.device("cpu")
    print("CUDA device is not available")
# %%
