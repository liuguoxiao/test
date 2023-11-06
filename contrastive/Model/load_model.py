import torch
import tqdm
import torch.nn as nn
import torchmetrics
from data_processing.mutation_data_get import get_forward_backward_mutation_data
from Dataloader.dataloader import GetLoader_load, GetValidLoader
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load(model_name, writer):
    model = torch.load("Model/pretrained_model.pth")
    # 数据集加载
    diff_anchor_result, diff_change_result, \
        diff_unchange_result, valid_diff_result, new_flag_result = get_forward_backward_mutation_data(
        is_model="load_model")
    train_dataloader = GetLoader_load(diff_change_result)
    train_dataloader = DataLoader(dataset=train_dataloader, batch_size=32, shuffle=True)
    valid_dataloader = GetValidLoader(valid_diff_result, new_flag_result)
    valid_dataloader = DataLoader(dataset=valid_dataloader, batch_size=32)
    if model_name == "GRU":
        # 冻结模型参数
        # for submodule in [model.gru, model.gru_c1]:
        #     for param in submodule.parameters():
        #         param.requires_grad = False
        optimizer = torch.optim.Adam([
            {'params': [p for sublist in [model.gru.parameters(), model.gru_c1.parameters()] for p in sublist if
                        p.requires_grad], 'lr': 1e-5},
            {'params': filter(lambda p: p.requires_grad, model.gru_c2.parameters()), 'lr': 1e-4}
        ])
        # optimizer = torch.optim.Adam(model.gru_c2.parameters(), lr=1e-3)
    elif model_name == "MLP":
        optimizer = torch.optim.Adam([
            {'params': [p for sublist in [model.mlp_layer.parameters()] for p in sublist if
                        p.requires_grad], 'lr': 1e-4},
            {'params': filter(lambda p: p.requires_grad, model.mlp_layer_c1.parameters()), 'lr': 5e-4}
        ])
    elif model_name == "CNN_LSTM":
        optimizer = torch.optim.Adam([
            {'params': [p for sublist in [model.conv1.parameters(), model.conv2.parameters(), model.conv2.parameters(),
                                          model.cnn_lstm.parameters(), model.gru_c1.parameters()] for p in sublist if
                        p.requires_grad], 'lr': 1e-5},
            {'params': filter(lambda p: p.requires_grad, model.gru_c2.parameters()), 'lr': 1e-4}
        ])

    criterion = nn.L1Loss(reduce=True, size_average=True, reduction="mean").to(device)

    for epoch in range(200):
        str_code = "train"
        model.train()
        data_iter = tqdm.tqdm(enumerate(train_dataloader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(train_dataloader),
                              bar_format="{l_bar}{r_bar}")
        for i, data in data_iter:
            diff_change_result = data.to(torch.float32)

            x, y, h = model.forward(diff_change_result[:, :, -1:].to(device))  # [:, :, -1:]
            optimizer.zero_grad()
            loss = criterion(x, y).to(device)
            loss.backward()
            optimizer.step()
            writer.add_scalar('load_model_loss/batch', loss.item(), epoch * len(train_dataloader) + i)
            writer.add_scalar('load_model_loss/epoch', loss, epoch)

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
                        x, y, h = model.forward(new_diff_before_data[:, :, -1:].to(device))  # [:, :, -1:]
                        # step_for_next.append(x)
                        step_for_next.append(x)
                        # store = step_for_next
                        store = torch.stack(step_for_next)
                        store = store.squeeze().unsqueeze(0)
                    else:
                        new_diff_before_data = diff_before_data[k, :, :].unsqueeze(0)
                        new_diff_before_data[:, :-1, -1][:, -store.numel():] = store
                        x, y, h = model.forward(new_diff_before_data[:, :, -1:].to(device))  # [:, :, -1:]
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
                    writer.add_scalar('load_model_valid/{}/batch'.format(name), metric_loss,
                                      epoch * len(valid_dataloader) + j)
                for name, metric in sudden_metrics.items():
                    metric_loss = metric(sudden_x, sudden_y)
                    writer.add_scalar('load_model_sudden_valid/{}/batch'.format(name), metric_loss,
                                      epoch * len(valid_dataloader) + j)
                for name, metric in un_sudden_metrics.items():
                    metric_loss = metric(un_sudden_x, un_sudden_y)
                    writer.add_scalar('load_model_un_sudden_valid/{}/batch'.format(name), metric_loss,
                                      epoch * len(valid_dataloader) + j)
        for name, metric_e in metrics.items():
            metric_epoch = metric_e.compute()
            writer.add_scalar('load_model_valid/{}/epoch'.format(name), metric_epoch, epoch)
            metric_e.reset()

        r2 = torchmetrics.R2Score().to(device)(torch.tensor(sudden_running_pres),
                                               torch.tensor(sudden_running_labels))
        writer.add_scalar('load_model_sudden_valid/r2/epoch', r2, epoch)
        un_sudden_r2 = torchmetrics.R2Score().to(device)(torch.tensor(un_sudden_running_pres),
                                                         torch.tensor(un_sudden_running_labels))
        writer.add_scalar('load_model_un_sudden_valid/r2/epoch', un_sudden_r2, epoch)
        for name, metric_e in sudden_metrics.items():
            metric_epoch = metric_e.compute()
            writer.add_scalar('load_model_sudden_valid/{}/epoch'.format(name), metric_epoch, epoch)
            metric_e.reset()
        for name, metric_e in un_sudden_metrics.items():
            metric_epoch = metric_e.compute()
            writer.add_scalar('load_model_un_sudden_valid/{}/epoch'.format(name), metric_epoch, epoch)
            metric_e.reset()
