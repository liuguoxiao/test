import torch
from torch.utils.data import DataLoader


# 定义GetLoader类，继承Dataset方法，并重写__getitem__()和__len__()方法
class GetLoader(torch.utils.data.Dataset):
    # 初始化函数，得到数据
    def __init__(self, diff_anchor_result, diff_change_result, diff_unchange_result):
        self.length = diff_anchor_result.shape[0]
        self.train_anchor_dataset = diff_anchor_result
        self.train_change_dataset = diff_change_result
        self.train_unchange_dataset = diff_unchange_result

    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        train_anchor_dataset = self.train_anchor_dataset[index].copy()
        diff_change_dataset = self.train_change_dataset[index].copy()
        train_unchange_dataset = self.train_unchange_dataset[index].copy()
        return train_anchor_dataset, diff_change_dataset, train_unchange_dataset

    # 该函数返回数据大小长度，目的是DataLoader方便划分
    def __len__(self):
        return self.length


class GetValidLoader(torch.utils.data.Dataset):
    # 初始化函数，得到数据
    def __init__(self, valid_diff_result, new_flag_result):
        self.length = valid_diff_result.shape[0]
        self.valid_data_dataset = valid_diff_result
        self.flag = new_flag_result

    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        valid_data_dataset = self.valid_data_dataset[index].copy()
        flag = self.flag[index].copy()
        return valid_data_dataset, flag

    # 该函数返回数据大小长度，目的是DataLoader方便划分
    def __len__(self):
        return self.length


class GetLoader_load(torch.utils.data.Dataset):
    # 初始化函数，得到数据
    def __init__(self, diff_change_result):
        self.length = diff_change_result.shape[0]
        self.train_change_dataset = diff_change_result

    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        diff_change_dataset = self.train_change_dataset[index].copy()
        return diff_change_dataset

    # 该函数返回数据大小长度，目的是DataLoader方便划分
    def __len__(self):
        return self.length
