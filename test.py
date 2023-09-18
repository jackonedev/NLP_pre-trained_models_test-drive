import torch

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print ("device: ", torch.cuda.get_device_name(0))

"""
[output]
True
1
device:  NVIDIA GeForce GT 1030
"""