from model_origin import HSIDCNN
import torch

#torch.cuda.set_device(0)

net = HSIDCNN()
#print(net)

y=torch.randn(12,1,36,20,20)
x=torch.randn(12,1,20,20)
out=net(x,y)
print(out.size())