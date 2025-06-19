from libpyCauchyKesai import CauchyKesai, __version__
print(__version__)
import numpy as np
# import torch
m = CauchyKesai("/app/model/basic/resnet18_224x224_nv12.hbm")


y = np.random.rand(1,224,224,1).astype(np.uint8)
uv = np.random.rand(1,112,112,1).astype(np.uint8)

o = m([y, uv])[0]
o.shape


m([y, torch.from_numpy(uv)])[0].shape