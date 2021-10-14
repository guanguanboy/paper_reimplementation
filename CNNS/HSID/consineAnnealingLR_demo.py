import torch
from torchvision.models import AlexNet
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt

model = AlexNet(num_classes=2)
optimizer = torch.optim.Adam(model.parameters(),lr=0.1)
scheduler = CosineAnnealingLR(optimizer,T_max=100)
plt.figure()
x = list(range(100))
y = []
for epoch in range(1,101):
    optimizer.zero_grad()
    optimizer.step()
    print("第%d个epoch的学习率：%f" % (epoch,optimizer.param_groups[0]['lr']))
    scheduler.step()
    y.append(scheduler.get_lr()[0])

# 画出lr的变化    
plt.plot(x, y)
plt.xlabel("epoch")
plt.ylabel("lr")
plt.title("learning rate's curve changes as epoch goes on!")
plt.show()
