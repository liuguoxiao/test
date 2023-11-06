import torch.nn as nn
import torch.nn.functional as D
import tqdm
import torchmetrics
import random
import yaml
import pandas as pd
import torch
import argparse
from torch.utils.data import DataLoader
from data_processing.mutation_data_get import get_forward_backward_mutation_data
from Train_Valid_Test_dataset.train_valid_test import train_valid_test
from Dataloader.dataloader import GetLoader, GetValidLoader
import os
from torch.utils.tensorboard import SummaryWriter
from Model.model import Model
import numpy as np
import logging
from Model.load_model import load

with open('config.yaml', 'r', encoding='utf-8') as F:
    config_yaml = yaml.safe_load(F)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=config_yaml['Batch']['batch_size'], type=int)
    parser.add_argument("--train_ratio", default=0.6, type=int)
    parser.add_argument("--valid_ratio", default=0.3, type=int)
    parser.add_argument('--log_dir', default='./log', help="the log directory for tensorboard ")
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--model', default=config_yaml['Model']['option'], type=str)
    parser.add_argument('--hidden_size', default=32, type=int)
    parser.add_argument('--num_layers', default=3, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)  # lstm 1e-3/5e-4;transformer 1e-4;mlp 1e-4
    parser.add_argument('--seed', default=1220, type=int, help='seed for initializing training')
    args = parser.parse_args()
    # 创建一个logger
    logger = logging.getLogger('example_logger')
    logger.setLevel(logging.INFO)
    file_name_variable = "model_{}".format(args.model)
    # 创建一个handler，用于写入日志文件
    fh = logging.FileHandler(f'{file_name_variable}.log', mode='w')  # 更新日志
    # fh = logging.FileHandler(f'{file_name_variable}.log')  # 追加日志
    # 定义handler的输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # 给logger添加handler
    logger.addHandler(fh)

    diff_anchor_result, diff_change_result, \
        diff_unchange_result, valid_diff_result, new_flag_result = get_forward_backward_mutation_data(is_model="model")
    # diff_before_result, diff_flip_after_result = get_forward_backward_mutation_data()
    # train_before_data_dataset, train_after_data_dataset, \
    #     valid_before_data_dataset, valid_after_data_dataset, \
    #     test_before_data_dataset, test_after_data_dataset = train_valid_test(diff_before_result,
    #                                                                          diff_flip_after_result,
    #                                                                          args.train_ratio,
    #                                                                          args.valid_ratio)
    train_dataloader = GetLoader(diff_anchor_result, diff_change_result, diff_unchange_result)
    train_dataloader = DataLoader(dataset=train_dataloader, batch_size=128, shuffle=True)
    valid_dataloader = GetValidLoader(valid_diff_result, new_flag_result)
    valid_dataloader = DataLoader(dataset=valid_dataloader, batch_size=args.batch_size)
    # test_dataloader = GetLoader(test_before_data_dataset, test_after_data_dataset)
    # test_dataloader = DataLoader(dataset=test_dataloader, batch_size=args.batch_size, shuffle=True)
    log_dir = args.log_dir + '/model_{}'.format(args.model) + '/epoch_{}+batch_{}+hidden_{}'.format(
        args.epochs, args.batch_size, args.hidden_size)
    writer = SummaryWriter(log_dir=log_dir)
    ckpt_dir = log_dir + '/checkpoints/'
    os.makedirs(ckpt_dir, exist_ok=True)
    model = Model(args.model, diff_change_result, args.hidden_size, args.num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss(reduce=True, size_average=True, reduction="mean").to(device)
    print("Training Start!")
    best_mae = float('inf')
    best_epoch = -1
    logger.info('-' * 50)
    logger.info('epochs: %d', args.epochs)
    for epoch in range(args.epochs):
        str_code = "train"
        model.train()
        data_iter = tqdm.tqdm(enumerate(train_dataloader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(train_dataloader),
                              bar_format="{l_bar}{r_bar}")
        for i, data in data_iter:
            train_anchor_dataset, diff_change_result, train_unchange_dataset = data[0].to(torch.float32), \
                data[1].to(torch.float32), data[2].to(torch.float32)
            anchor_x, anchor_y, anchor_h = model.forward(train_anchor_dataset[:, :, -1:].to(device))  # [:, :, -1:]
            x, y, h = model.forward(diff_change_result[:, :, -1:].to(device))  # [:, :, -1:]
            un_x, un_y, u_h = model.forward(train_unchange_dataset[:, :, -1:].to(device))  # [:, :, -1:]
            optimizer.zero_grad()
            d_positive = distance_tensor(anchor_h, h)
            d_negative = distance_tensor(anchor_h, u_h)
            a = anchor_h.mean()
            b = h.mean()
            c = u_h.mean()
            d = d_positive.mean()
            e = d_negative.mean()
            f = (4 + d_positive - d_negative).min()
            g = (4 + d_positive - d_negative).max()
            loss = torch.clamp(3 + d_positive - d_negative, min=0.0).mean()  # 0.04  GRU:4 MLP:2
            # loss = D.cosine_similarity(h, u_h).mean()
            # loss = torch.clamp(1/p_n, min=0.5).mean()
            # sudden_loss = criterion(x, y).to(device)
            # unsudden_loss = criterion(un_x, un_y).to(device)
            # loss = sudden_loss / (sudden_loss + unsudden_loss) * sudden_loss
            loss.backward()
            optimizer.step()
            writer.add_scalar('loss/batch', loss.item(), epoch * len(train_dataloader) + i)
            writer.add_scalar('loss/epoch', loss, epoch)

        # validation
        model.eval()
        metrics = {
            'MSE': torchmetrics.MeanSquaredError().to(device),
            'MAE': torchmetrics.MeanAbsoluteError().to(device),
            'R2': torchmetrics.R2Score().to(device),
            'MAPE': torchmetrics.MeanAbsolutePercentageError().to(device)
        }
        sudden_metrics = {
            'MSE': torchmetrics.MeanSquaredError().to(device),
            'MAE': torchmetrics.MeanAbsoluteError().to(device),
            # 'R2': torchmetrics.R2Score().to(device),
            'MAPE': torchmetrics.MeanAbsolutePercentageError().to(device)
        }
        un_sudden_metrics = {
            'MSE': torchmetrics.MeanSquaredError().to(device),
            'MAE': torchmetrics.MeanAbsoluteError().to(device),
            # 'R2': torchmetrics.R2Score().to(device),
            'MAPE': torchmetrics.MeanAbsolutePercentageError().to(device)
        }
        data_iter = tqdm.tqdm(enumerate(valid_dataloader),
                              total=len(valid_dataloader),
                              bar_format="{l_bar}{r_bar}")
        running_pres = []
        running_labels = []
        sudden_running_pres = []
        sudden_running_labels = []
        un_sudden_running_pres = []
        un_sudden_running_labels = []
        zero_one_flag = []
        with torch.no_grad():
            for j, valid_data in data_iter:
                step_for_next = []
                for k in range(valid_data[0].shape[0]):
                    if k == 0:
                        diff_before_data, flag = valid_data[0].to(torch.float32), \
                            valid_data[1].to(torch.float32)
                        new_diff_before_data = diff_before_data[k, :, :].unsqueeze(0)
                        x, y, h = model.forward(new_diff_before_data[:, :, -1:].to(device))#[:, :, -1:]
                        # step_for_next.append(x)
                        step_for_next.append(x)
                        # store = step_for_next
                        store = torch.stack(step_for_next)
                        store = store.squeeze().unsqueeze(0)
                    else:
                        new_diff_before_data = diff_before_data[k, :, :].unsqueeze(0)
                        new_diff_before_data[:, :-1, -1][:, -store.numel():] = store
                        x, y, h = model.forward(new_diff_before_data[:, :, -1:].to(device))#[:, :, -1:]
                        step_for_next.append(x)
                        # store = step_for_next
                        store = torch.stack(step_for_next)
                        store = store.squeeze().unsqueeze(0)
                x = store.t().to(device)
                y = diff_before_data[:, -1:, -1].to(device)
                running_pres.extend(x.detach().cpu().numpy().ravel())
                running_labels.extend(y.detach().cpu().numpy().ravel())
                # if j == 0:
                #     diff_before_data, diff_after_data, flag = valid_data[0].to(torch.float32), \
                #         valid_data[1].to(torch.float32), valid_data[2].to(torch.float32)
                #     x, y = model.forward(diff_before_data.to(device), diff_after_data.to(device))
                sudden_x = x[flag == 1]
                sudden_y = y[flag == 1]
                sudden_running_pres.extend(sudden_x.detach().cpu().numpy().ravel())
                sudden_running_labels.extend(sudden_y.detach().cpu().numpy().ravel())
                un_sudden_x = x[flag == 0]
                un_sudden_y = y[flag == 0]
                un_sudden_running_pres.extend(un_sudden_x.detach().cpu().numpy().ravel())
                un_sudden_running_labels.extend(un_sudden_y.detach().cpu().numpy().ravel())
                zero_one_flag.extend(flag.detach().cpu().numpy().ravel())
                for name, metric in metrics.items():
                    metric_loss = metric(x, y)
                    writer.add_scalar('valid/{}/batch'.format(name), metric_loss,
                                      epoch * len(valid_dataloader) + j)
                for name, metric in sudden_metrics.items():
                    metric_loss = metric(sudden_x, sudden_y)
                    writer.add_scalar('sudden_valid/{}/batch'.format(name), metric_loss,
                                      epoch * len(valid_dataloader) + j)
                for name, metric in un_sudden_metrics.items():
                    metric_loss = metric(un_sudden_x, un_sudden_y)
                    writer.add_scalar('un_sudden_valid/{}/batch'.format(name), metric_loss,
                                      epoch * len(valid_dataloader) + j)
        for name, metric_e in metrics.items():
            metric_epoch = metric_e.compute()
            writer.add_scalar('valid/{}/epoch'.format(name), metric_epoch, epoch)
            metric_e.reset()

        r2 = torchmetrics.R2Score().to(device)(torch.tensor(sudden_running_pres), torch.tensor(sudden_running_labels))
        writer.add_scalar('sudden_valid/r2/epoch', r2, epoch)
        un_sudden_r2 = torchmetrics.R2Score().to(device)(torch.tensor(un_sudden_running_pres),
                                                         torch.tensor(un_sudden_running_labels))
        writer.add_scalar('un_sudden_valid/r2/epoch', un_sudden_r2, epoch)
        for name, metric_e in sudden_metrics.items():
            metric_epoch = metric_e.compute()
            if name == 'MAE' and metric_epoch < best_mae:
                best_mae = metric_epoch
                best_epoch = epoch
                logger.info('best_epoch: %d', best_epoch)
                logger.info('best_mae: %f', best_mae)
                df = pd.DataFrame({'running_pres': np.array(running_pres),
                                   'running_labels': np.array(running_labels),
                                   'zero_one_flag': np.array(zero_one_flag)
                                   })
                df.to_csv('best_epoch/{}_best_predictions.csv'.format(args.model), index=False)
                sudden_df = pd.DataFrame({'running_pres': np.array(sudden_running_pres),
                                          'running_labels': np.array(sudden_running_labels)
                                          })
                sudden_df.to_csv('best_epoch/{}_sudden_best_predictions.csv'.format(args.model), index=False)
            writer.add_scalar('sudden_valid/{}/epoch'.format(name), metric_epoch, epoch)
            metric_e.reset()
        for name, metric_e in un_sudden_metrics.items():
            metric_epoch = metric_e.compute()
            writer.add_scalar('un_sudden_valid/{}/epoch'.format(name), metric_epoch, epoch)
            metric_e.reset()

    # 保存预训练模型的参数
    torch.save(model, 'Model/pretrained_model.pth')
    load(args.model, writer)

    writer.close()
    data_for_accuracy = pd.read_csv('best_epoch/{}_best_predictions.csv'.format(args.model))
    pres = data_for_accuracy['running_pres'].values
    label = data_for_accuracy['running_labels'].values
    flag = data_for_accuracy['zero_one_flag'].values
    pres_output = zero_one_ratio(pres)
    label_output = zero_one_ratio(label)
    equal_elements = (flag == label_output)
    overlap = np.mean(equal_elements)
    overlap_ones = np.logical_and(pres_output == 1, label_output == 1)
    # overlap_ones = np.logical_and(pres_output == 1, flag == 1)
    # 计算重合率，即两个数组相同位置都为1的数量除以两个数组中1的总数
    overlap_rate = np.sum(overlap_ones) / np.sum(label_output == 1)
    logger.info('overlap_rate: %f', overlap_rate)
    print("Training END")


def zero_one_ratio(arr):
    segment_length = config_yaml['Batch']['batch_size']
    threshold = config_yaml['AQI_mutation']['AQI_threshold']
    window_length = config_yaml['AQI_mutation']['AQI_window_size'] - 1
    result = []
    segments = [arr[i:i + segment_length] for i in range(0, len(arr), segment_length)]

    for segment in segments:
        for i in range(len(segment)):
            abs = int(np.abs(segment[i]) * 500)
            sum_one_abs = int(np.abs(sum(segment[i:i + 1])) * 500)
            sum_two_abs = int(np.abs(sum(segment[i:i + 2])) * 500)
            sum_three_abs = int(np.abs(sum(segment[i:i + 3])) * 500)
            if abs >= threshold or sum_one_abs >= threshold or sum_two_abs >= threshold:
                result.append(1)
            else:
                result.append(0)
    result = np.array(result)
    return result


def distance_tensor(x1, x2):
    diff = torch.abs(x1 - x2)
    return torch.pow(diff, 2).sum(dim=1)
