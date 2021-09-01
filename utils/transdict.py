import torch
from collections import OrderedDict
b = OrderedDict()

name = "checkpoint/vit_L16_224/epoch_7.pth"
a = torch.load(name)
for k in a.keys():
    b[k.replace("module.", "")] = a[k]

torch.save(b, "vit_L16_224_90.2.pth")

