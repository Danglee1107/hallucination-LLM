import torch

print(torch.__version__, torch.version.cuda)
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print(torch.cuda.get_arch_list())
