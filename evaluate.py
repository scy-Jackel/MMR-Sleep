import matplotlib.pyplot as plt
from sleepdata.dataset import *
from model_ulit.metric import class_metric
from model_ulit.model import NEWNet
from function import cost_time
import pandas as pd
import torch
import torch.nn as nn
@cost_time
def main_evaluate(config, args):
    batch_size = config.config["data_loader"]["args"]["batch_size"]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #读取测试集
    test_dataset = sleppEDF(root_path_normal=args.np_data_dir, flag='val',
                                      split=config.config["data_set"]["sleepedf_split"]
                                      )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    #读取模型
    model = NEWNet(test_dataset.n_samples, test_dataset.sfreq,
                   len(test_dataset.channels), config.config["model"],test_dataset.type_n).to(device)
    model.load_state_dict(torch.load(config.config["trainer"]["save_dir"] + "/model/" + str(
        config.config["data_set"]["sleepedf_split"][1]) + "best_model_f1_all.pth"))
    model = model.to(device)
    output_list = []
    target_list = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target1) in enumerate(test_loader):
            data, target1 = data.to(device), target1.to(device)
            output1 = model(data)
            if output_list == []:
                output_list = output1
                target_list = target1
            else:
                output_list = torch.cat((output_list, output1), dim=0)
                target_list = torch.cat((target_list, target1), dim=0)
    class_acc, class_precision, class_recall, class_F1, class_distribution, Mf1 = class_metric(output_list, target_list)
    #写入excel文件
    torch.set_printoptions(precision=4, sci_mode=False)
    with open(config.config["trainer"]["save_dir"] + '/classfier_result/'+str(
                               config.config["data_set"]["sleepedf_split"][1])+'.txt', 'w') as f:
        f.write('对于正常信息，78个数据中的测试，表现如下 ' + '\n')
        #保留四位小数写入
        f.write('准确率：' + str(class_acc) + '\n')
        f.write('精确率：' + str(class_precision) + '\n')
        f.write('召回率：' + str(class_recall) + '\n')
        f.write('F1值：' + str(class_F1) + '\n')
        #混淆矩阵以整数写入,不使用科学计数法
        f.write('混淆矩阵：' + '\n')
        f.write(str(class_distribution) + '\n')
        #写入Mf1
        f.write('Mf1：' + str(Mf1) + '\n')
    #重新写入模型
    torch.save(model.state_dict(),
               config.config["trainer"]["save_dir"] + "/model/" + str(
                   config.config["data_set"]["sleepedf_split"][1]) + "best_model_f1_all.pth")

