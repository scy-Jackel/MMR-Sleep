from model_ulit.loss import trainLoss, evaLoss
from model_ulit.model import NEWNet
from function import cost_time
from sleepdata.dataset import *
import torch
from model_ulit.metric import class_metric
from tqdm import tqdm


@cost_time
def main_train(config, args):
    best_f1 = 0
    patience_max = 6
    batch_size = config.config["data_loader"]["args"]["batch_size"]

    # 如果是gpu就用第一个gpu，如果是cpu就用cpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    val_dataset = sleppEDF(args.np_data_dir, 'val', config.config["data_set"]["sleepedf_split"]
                           )
    # 加载数据集，split_num代表第几次划分数据集，split_num=1代表第一次划分数据集，数据分批导入，减少内存占用
    trian_dataset = sleppEDF(args.np_data_dir, 'train', config.config["data_set"]["sleepedf_split"]
                             )

    type_n = int(trian_dataset.type_n)
    model = NEWNet(trian_dataset.n_samples, trian_dataset.sfreq,
                   len(trian_dataset.channels), config.config["model"], type_n).to(device)
    train_loader = torch.utils.data.DataLoader(trian_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size * 4, shuffle=False, num_workers=0)
    # 指定损失函数，优化器，评价指标
    loss_train = trainLoss(type_n, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.config["optimizer"]["args"]["lr"],
                                  weight_decay=config.config["optimizer"]["args"]["weight_decay"])

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.10)
    change_ed = 0
    loss_plot = []
    val_loss_plot = []
    min_loss = 100
    patience = patience_max
    # 训练
    for epoch in range(1, int((config.config["trainer"]["epochs"])) + 1):
        model.train()
        loss_sum = 0

        # 输出当前epoch和这个epoch的学习率,batch数量
        print("epoch:", epoch, "lr:", optimizer.param_groups[0]["lr"], "batch_num:", len(train_loader))
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        train_len = len(train_loader)
        for batch_idx, (data, target) in loop:
            # 将数据放到gpu上
            data, target = data.to(device), target.to(device)
            # 情空梯度
            optimizer.zero_grad()
            output, x = model(data, model="train")
            loss = loss_train(output, target, x=x)
            loss.backward()
            optimizer.step()
            loop.set_description(f'Epoch [{epoch}/{int((config.config["trainer"]["epochs"]))}]')
            loop.set_postfix(loss=loss.item())
            loss_sum += loss.item()
        val_loss_mean, MF1, class_F1, best_f1, patience, min_loss = evaluate(model, val_loader, type_n, device,
                                                                             best_f1, min_loss, patience,
                                                                             patience_max, config, epoch)
        # 计算验证集的损失
        lr_scheduler.step()
        loss_mean = loss_sum / len(train_loader)
        loss_plot.append(loss_mean)
        val_loss_plot.append(val_loss_mean)
        with open(config.config["trainer"]["save_dir"] + "/loss/" + str(
                config.config["data_set"]["sleepedf_split"][1]) + "loss.txt", "a") as f:
            if epoch == 1:
                # 写入一行横向作为分割
                f.write("-------------------------新的训练开始了-------------------------------" + "\n")
            f.write("epoch:" + str(epoch) + " loss:" + str(loss_mean) + " val_loss:"
                    + str(val_loss_mean) + "Mf1_val" + str(MF1) +
                    "min_loss" + str(min_loss) + "class_f1" + str(class_F1) + "\n")
            # 如果是最后一个epoch写入loss_plot
            if epoch == int((config.config["trainer"]["epochs"])):
                f.write("loss_plot:" + str(loss_plot) + "\n")
                # 写入val_loss_plot
                f.write("val_loss_plot:" + str(val_loss_plot) + "\n")
        # 每个epoch结束后，打印loss，val_loss,并将val_loss和loss保存到列表中，存入一个txt文件用于画图,小数点后保留4位
        print("\r", "    ", "epoch:", epoch, "loss:", round(loss_mean, 4), "val_loss:", round(val_loss_mean, 4),
              "f1_avelue:", round(MF1, 4),
              "class_f1:", [round(i, 4) for i in class_F1], "\n", "best_f1:", round(best_f1, 4),
              "min_loss:", round(min_loss, 4), "patience:",
              patience)
        # 打印一行横向
        print("--------------------------------------------------------")
        if patience <= 0:
            # 如果没有提升就停止训练，并写入最后的loss_plot和val_loss_plot
            with open(config.config["trainer"]["save_dir"] + "/loss/" + str(
                    config.config["data_set"]["sleepedf_split"][1]) + "loss.txt", "a") as f:
                f.write("loss_plot:" + str(loss_plot) + "\n")
                # 写入val_loss_plot
                f.write("val_loss_plot:" + str(val_loss_plot) + "\n")
            break
    return best_f1


def evaluate(model, val_loader, type_n, device, best_f1, min_loss, patience, patience_max, config, epoch):
    val_loss_sum = 0
    output_list = []
    target_list = []
    loss_val = evaLoss(type_n).to(device)
    with torch.no_grad():
        model.eval()
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            # 将negtive的第一个维度和第二个维度合并
            output = model(data)
            val_loss = loss_val(output, target)
            if output_list == []:
                output_list = output
                target_list = target
            else:
                output_list = torch.cat((output_list, output), dim=0)
                target_list = torch.cat((target_list, target), dim=0)
            val_loss_sum += val_loss.item()
        _, _, _, class_F1, _, _ = class_metric(output_list, target_list)

    MF1 = sum(class_F1) / len(class_F1)
    val_loss_mean = val_loss_sum / len(val_loader)
    # 计算list:class_f1均值
    # 保存损失值最小的模型
    if val_loss_mean < min_loss:
        min_loss = val_loss_mean
        torch.save(model.state_dict(),
                   config.config["trainer"]["save_dir"] + "/model/" + str(
                       config.config["data_set"]["sleepedf_split"][1])
                   + "best_model.pth")
        patience = patience_max
        print("min_loss:", min_loss, "保存完毕")
    if MF1 > best_f1:
        best_f1 = MF1
        torch.save(model.state_dict(),
                   config.config["trainer"]["save_dir"] + "/model/" + str(
                       config.config["data_set"]["sleepedf_split"][1])
                   + "best_model_f1_all.pth")
        print("best_f1:", best_f1, "保存完毕", "MF1:", MF1, "class_F1:", class_F1)
        patience = patience_max
    if val_loss_mean > min_loss and MF1 < best_f1:
        patience -= 1
    return val_loss_mean, MF1, class_F1, best_f1, patience, min_loss
