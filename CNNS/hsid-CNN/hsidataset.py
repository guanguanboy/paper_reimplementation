import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

from os import listdir
from os.path import join
import scipy.io as scio

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.mat'])

class HsiTrainDataset(Dataset):
    def __init__(self, dataset_dir):
        super(HsiTrainDataset, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        mat = scio.loadmat(self.image_filenames[index], verify_compressed_data_integrity=False)
        noisy = mat['noisy'].astype(np.float32)
        label = mat['label'].astype(np.float32)

        # 增加一个维度，因为HSID模型处理的是四维tensor，因此这里不需要另外增加一个维度
        # noisy_exp = np.expand_dims(noisy, axis=0)
        # label_exp = np.expand_dims(label, axis=0)

        return torch.from_numpy(noisy), torch.from_numpy(label)

    def __len__(self):
        return len(self.image_filenames)

def run_dataset_test():
    batch_size = 1
    #train_set = HsiTrainDataset('./data/train/')
    train_set = HsiTrainDataset('./data/test/')
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    print(next(iter(train_loader))[0].shape)
    print(next(iter(train_loader))[1].shape)
    print(len(train_loader))

#run_dataset_test()