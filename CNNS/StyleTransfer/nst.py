import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

model = models.vgg19(pretrained=True).features
print(model)

# ['0', '5', '10', '19', '28']

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        self.chosen_features = ['0', '5', '10', '19', '28']
        self.model = models.vgg19(pretrained=True).features[:29]

    def forward(self, x):
        features = []

        for layer_num, layer in enumerate(self.model):
            x = layer(x)

            if str(layer_num) in self.chosen_features:
                features.append(x) #将用来计算的特征保存到features中

        return features

def load_image(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 356

loader = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ]
)

original_img = load_image("annahathaway.png")
style_img = load_image("style.jpg")

model = VGG().to(device).eval()

generated = original_img.clone().requires_grad_(True)

# Hyperparameters
total_steps = 6000
learning_rate = 0.001
alpha = 1
beta = 0.01 # cotrols how much sytle do we want from the image
optimizer = optim.Adam([generated], lr=learning_rate)  #注意这里优化的不是模型的权重，generated的像素值

for step in range(total_steps):
    generated_features = model(generated) #将三张图片分别通过VGG网络提取特征
    original_img_features = model(original_img)
    style_features = model(style_img)

    style_loss = original_loss = 0

    for gen_features, orig_feature, style_features in zip(
        generated_features, original_img_features, style_features
    ):
        batch_size, channel, height, width = gen_features.shape
        original_loss += torch.mean((gen_features - orig_feature) ** 2)

        #compute gram matrix #由于batch_size为1，我们只加载进来了一张照片
        G = gen_features.view(channel, height*width).mm(
            gen_features.view(channel, height*width).t()
        )

        A = style_features.view(channel, height*width).mm(
            style_features.view(channel, height*width).t()
        )

        style_loss += torch.mean((G - A)**2)

    total_loss = alpha*original_loss + beta * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(total_loss)
        save_image(generated, "generated.png")





