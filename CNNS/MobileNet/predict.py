import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model_v2 import MobileNetV2


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    

    # load image
    img_path = "../tulip.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path) #载入图像
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img) ##使用和预处理过程中相同的图像预处理方式
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0) #添加一个batch维度

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = MobileNetV2(num_classes=5).to(device)
    # load model weights
    model_weight_path = "./MobileNetV2.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device)) #载入迁移学习之后训练好的模型权重
    model.eval() #进入eval模式
    with torch.no_grad(): #通过这句话，禁止运算过程中跟踪我们的误差梯度信息，从而节省内存空间
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu() #将图像传入模型当中
        predict = torch.softmax(output, dim=0) #将输出转换成概率分布
        predict_cla = torch.argmax(predict).numpy() #获得最大的预测概率值所对应的索引

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    print(print_res)
    plt.show()


if __name__ == '__main__':
    main()