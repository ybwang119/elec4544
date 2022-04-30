from thop import profile
import torch
from models.alexnet_miniImage_gr_shuffle import Alexnet_miniImage_gr_shuffle
flower = torch.randn(1, 3, 84, 84)
model=Alexnet_miniImage_gr_shuffle(100)
flops, params = profile(model, inputs=(flower, ))
print(flops)
print(params)