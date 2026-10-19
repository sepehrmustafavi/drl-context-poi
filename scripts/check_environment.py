import json
import torch
from drlpoi.util import environment

x = environment()
x["visible_gpu_count"] = torch.cuda.device_count()
print(json.dumps(x, indent=2))
if torch.cuda.is_available():
    a = torch.ones(8, 8, device="cuda")
    print("CUDA tensor check:", (a @ a).sum().item())
