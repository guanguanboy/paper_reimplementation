import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

from os import listdir
from os.path import join
import scipy.io as scio

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.mat'])

class HsiCubicTrainDataset(Dataset):
    def __init__(self, dataset_dir):
        super(HsiCubicTrainDataset, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        mat = scio.loadmat(self.image_filenames[index], verify_compressed_data_integrity=False)
        noisy = mat['patch'].astype(np.float32)
        label = mat['label'].astype(np.float32)
        cubic = mat['cubic'].astype(np.float32)
        # 增加一个维度，因为HSID模型处理的是四维tensor，因此这里不需要另外增加一个维度
        noisy_exp = np.expand_dims(noisy, axis=0)
        label_exp = np.expand_dims(label, axis=0)
        #将36个band的cubic，去掉前三个，去掉后三个，变成30个band
        #noisy_cubic_30 = cubic[3:-3]
        noisy_cubic_exp = np.expand_dims(cubic, axis=0) #ENCAM网络第二个分支是3D卷积

        return torch.from_numpy(noisy_exp), torch.from_numpy(noisy_cubic_exp), torch.from_numpy(label_exp)

    def __len__(self):
        return len(self.image_filenames)

def run_cubic_train_dataset():
    batch_size = 1
    #train_set = HsiTrainDataset('./data/train/')
    test_set = HsiCubicTrainDataset('../HSID/data/train_cubic/')
    train_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
    print(next(iter(train_loader))[0].shape)
    print(next(iter(train_loader))[1].shape)
    print(len(train_loader))


class HsiCubicLowlightTestDataset(Dataset):
    def __init__(self, dataset_dir):
        super(HsiCubicLowlightTestDataset, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        dataset_dir_len = len(dataset_dir)
        self.image_filenames.sort(key = lambda x: int(x[dataset_dir_len:-4])) #升序排列文件名
        print(self.image_filenames)

    def __getitem__(self, index):
        mat = scio.loadmat(self.image_filenames[index], verify_compressed_data_integrity=False)
        noisy = mat['noisy'].astype(np.float32)
        label = mat['label'].astype(np.float32)
        noisy_cubic = mat['cubic'].astype(np.float32)
        # 增加一个维度，因为HSID模型处理的是四维tensor，因此这里不需要另外增加一个维度
        noisy_exp = np.expand_dims(noisy, axis=0)
        label_exp = np.expand_dims(label, axis=0)

        #将36个band的cubic，去掉前三个，去掉后三个，变成30个band
        #noisy_cubic_30 = noisy_cubic[3:-3]
        noisy_cubic_exp = np.expand_dims(noisy_cubic, axis=0) #ENCAM网络第二个分支是3D卷积

        return torch.from_numpy(noisy_exp), torch.from_numpy(noisy_cubic_exp), torch.from_numpy(label_exp)

    def __len__(self):
        return len(self.image_filenames)

def run_cubic_test_dataset():
    batch_size = 1
    #train_set = HsiTrainDataset('./data/train/')
    test_set = HsiCubicLowlightTestDataset('../HSID/data/test_lowlight/cubic/')
    train_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
    print(next(iter(train_loader))[0].shape)
    print(next(iter(train_loader))[1].shape)
    print(len(train_loader))

class HsiLowlightTestDataset(Dataset):
    def __init__(self, dataset_dir):
        super(HsiLowlightTestDataset, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        mat = scio.loadmat(self.image_filenames[index], verify_compressed_data_integrity=False)
        noisy = mat['lowlight'].astype(np.float32)
        label = mat['label'].astype(np.float32)

        # 增加一个维度，因为HSID模型处理的是四维tensor，因此这里不需要另外增加一个维度
        #noisy_exp = np.expand_dims(noisy, axis=0)
        #label_exp = np.expand_dims(label, axis=0)

        return torch.from_numpy(noisy), torch.from_numpy(label)

    def __len__(self):
        return len(self.image_filenames)

if __name__ == '__main__':
    run_cubic_train_dataset()
    #run_cubic_test_dataset()
