import torch
from collections import OrderedDict

a = torch.load("models_local/4head_norm_83.3_e20.pth")
al = list(a.keys())
b = OrderedDict()

for k in al:
    if 'backbone' in k and 'fc' not in k:
        b[k.replace("backbone.", "")] = a[k]

for i in range(4):
    b['classifier.multi_head.{}.head.weight'.format(i)] = a['classifier.fc{}.weight'.format(i+1)]
print(b.keys())

torch.save(b, "models_local/resnet101_coco_head4_lam0.5_83.3.pth")
