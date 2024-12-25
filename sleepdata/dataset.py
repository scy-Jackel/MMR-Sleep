import torch
import torch.nn as nn
import scipy
import numpy as np
import glob
import os


class sleppEDF(nn.Module):
    def __init__(self, root_path_normal, flag, split):
        # 记录有几种通道组合
        super(sleppEDF).__init__()
        self.root_path = root_path_normal
        self.flag = flag
        self.split = split
        self.fold_num = int(1 / (1 - self.split[0]))
        self.time_data, self.label, self.channels, self.n_len, self.sfreq, self.type_n, self.type_n_list = self.__read_data_train__()
        self.n_samples = self.time_data.shape[2]
        self.n_features = self.time_data.shape[1]

    def add_oversampling(self):
        save_name = self.root_path.split("\\")[-1] + ".pt"
        save_name_expand = save_name.replace("_npz.pt", "_expand.pt")
        save_path_f = os.path.join(self.root_path, str("readed"))
        save_path_expand = os.path.join(save_path_f, save_name_expand)
        data = torch.load(save_path_expand)
        data_x_expand, data_y_expand = data["x"], data["y"]
        data_x_expand = data_x_expand[:self.split[1] - 1] + data_x_expand[self.split[1]:]
        data_y_expand = data_y_expand[:self.split[1] - 1] + data_y_expand[self.split[1]:]
        data_x_expand = torch.vstack(data_x_expand)
        data_y_expand = torch.vstack(data_y_expand)
        self.time_data = torch.cat((self.time_data, data_x_expand), dim=0)
        self.label = torch.cat((self.label, data_y_expand), dim=0)
        self.n_len = self.time_data.shape[0]

    def __read_data_train__(self):

        save_name = self.root_path.split("\\")[-1] + ".pt"
        save_name = save_name.replace("_npz.pt", ".pt")
        save_path_f = os.path.join(self.root_path, str("readed"))
        save_path = os.path.join(save_path_f, save_name)
        if not os.path.exists(save_path_f):
            # 创建文件夹
            os.makedirs(save_path_f)
        # 读取原始数据，如果没有保存过，则保存
        if not os.path.exists(save_path):
            data_x, data_y, channels, freq, type_n = self.__save_data__(save_path)
        else:
            data = torch.load(save_path)
            data_x, data_y, channels, freq, type_n = data["x"], data["y"], data["channels"], \
                data["freq"], data["type_n"]
        # 读取扩充数据
        save_name = self.root_path.split("\\")[-1] + ".pt"
        save_name_expand = save_name.replace("_npz.pt", "_expand.pt")
        save_path_f = os.path.join(self.root_path, str("readed"))
        save_path_expand = os.path.join(save_path_f, save_name_expand)
        if not os.path.exists(save_path_expand):
            data_x_expand, data_y_expand = self.__over_samping__(self.time_data, self.label, self.type_n,
                                                                 save_path_expand)
        # 根据self.flag判断是训练集还是验证集,并返回对应的数据
        if self.flag == "train":
            # 去除验证集的编号self.split[1]
            data_x = data_x[:self.split[1] - 1] + data_x[self.split[1]:]
            data_y = data_y[:self.split[1] - 1] + data_y[self.split[1]:]
            data_x = torch.vstack(data_x)
            data_y = torch.vstack(data_y)
        else:
            # 只取验证集的编号self.split[1]
            pick_num = self.split[1] - 1
            data_x = data_x[pick_num]
            data_y = data_y[pick_num]
        # 统计每种类型的数据量并输出
        data_len = data_x.shape[0]
        type_n_list = []
        data_y_arg = torch.argmax(data_y, dim=1)
        for i in range(type_n):
            type_n_list.append(torch.sum(data_y_arg == i).item())
        print(self.flag, "type_n_list:", type_n_list)
        return data_x, data_y, channels, data_len, freq, type_n, type_n_list

    def __getitem__(self, index):
        return self.time_data[index, :, :], self.label[index]

    def __save_data__(self, save_path):
        # 查找有几个文件
        psg_fnames = glob.glob(os.path.join(self.root_path, "*.npz"))
        psg_fnames.sort()
        data_x = []
        data_y = []
        for file in range(len(psg_fnames)):
            data = np.load(psg_fnames[file])
            if file == 0:
                data_x = [torch.tensor(data["x"].transpose(1, 0, 2), dtype=torch.float32)]
                data_y = [torch.tensor(data["y"], dtype=torch.float32)]
                type_y = torch.unique(data_y[0])
                type_n = type_y.shape[0]
                data_y[0] = hot_y(data_y[0], type_n)
            else:
                data_x.append(torch.tensor(data["x"].transpose(1, 0, 2), dtype=torch.float32))
                data_y.append(hot_y(torch.tensor(data["y"], dtype=torch.float32), type_n))
        channels = data["ch_label"]
        freq = data["fs"]
        data_x = torch.vstack(data_x)
        data_y = torch.vstack(data_y)
        # 将data_x, data_y均分self.fold_num份
        data_x = torch.split(data_x, data_x.shape[0] // self.fold_num)
        data_y = torch.split(data_y, data_y.shape[0] // self.fold_num)
        # 保存x=data_x, y=data_y, channels=channels,data_len=data_len, freq=freq, type_n=type_n为.pt文件
        data = {"x": data_x, "y": data_y, "channels": channels, "freq": freq,
                "type_n": type_n}
        torch.save(data, save_path)
        return data_x, data_y, channels, freq, type_n

    def __over_samping__(self, data_x, data_y, type_n, save_path_expand):
        data_len = data_x[0].shape[2]
        data_x_expand = []
        data_y_expand = []
        for i in range(len(data_x)):
            data_x_i = []
            data_y_i = []
            # 寻找每种类型的数据的量
            type_n_list = torch.sum(data_y[i], dim=0)
            data_y_arg = torch.argmax(data_y[i], dim=1)
            mean_n = torch.mean(type_n_list).int()
            for j in range(type_n):
                if type_n_list[j] < mean_n:
                    place = torch.where(data_y_arg == j)[0]
                    for k in range(1, len(place)):
                        if place[k] - place[k - 1] == 1:
                            data_x_new = torch.cat((data_x[i][place[k - 1], :, (data_len // 5) * 3:],
                                                    data_x[i][place[k], :, 0:(data_len // 5) * 3]), dim=1).unsqueeze(0)
                            data_y_new = data_y[i][place[k]].unsqueeze(0)
                            data_x_i.append(data_x_new)
                            data_y_i.append(data_y_new)
                            type_n_list[j] += 1
                        if type_n_list[j] == mean_n:
                            break
            data_x_expand.append(torch.vstack(data_x_i))
            data_y_expand.append(torch.vstack(data_y_i))
        # 保存data_x_expand, data_y_expand为.pt文件
        data = {"x": data_x_expand, "y": data_y_expand}
        torch.save(data, save_path_expand)
        return data_x_expand, data_y_expand

    def __len__(self):
        return self.n_len


def hot_y(y, type_n):
    y_hot = torch.zeros(y.shape[0], type_n).to(y.device)
    place = torch.arange(y.shape[0], dtype=torch.long)
    y = y.to(torch.long)
    y_hot[place, y] = 1
    return y_hot
