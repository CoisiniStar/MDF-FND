import math
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os
import torch
import sys

sys.path.append('../')
from utils import config, UEM_utils
from torch.utils.data import DataLoader
from UEM.uem import DisTrans
from Engine import init_weights, TF, EF
from torch.optim import Adam
# import text_engine
from transformers import get_linear_schedule_with_warmup
from torch.utils.data.dataloader import default_collate
from MDFwithCS import MDFTransformer

from pytorch_model_summary import summary


# 自定义collate_fn函数
def collate_fn(batch):
    # 修改代码
    # 找到tensor1和tensor2的第一个维度的最大长度
    max_length_dim1 = max(item[0].shape[0] for item in batch)
    max_length_dim2 = max(item[1].shape[0] for item in batch)

    # 扩展tensor1和tensor2的第一个维度
    expanded_data = [
        (torch.cat([item[0], torch.zeros(max_length_dim1 - item[0].shape[0], item[0].shape[1])]),
         torch.cat([item[1], torch.zeros(max_length_dim2 - item[1].shape[0], item[1].shape[1])]),
         item[2])
        for item in batch
    ]
    return expanded_data


def set_requires_grad(model, require_grad=True):
    for param in model.parameters():
        param.requires_grad = require_grad


if __name__ == '__main__':
    # 设置dataset
    dataset_name = 'me15'
    UEM_utils.set_seed(5)
    if dataset_name == "me15":
        dataset_train, dataset_test = UEM_utils.set_up_mediaeval2015()
        print('Me15 Dataset')
        # dataset_train,dataset_test = textDataset.UEMDataset()
    elif dataset_name == "weibo":
        dataset_train, dataset_test = UEM_utils.set_up_weibo()
        print('Weibo Dataset')
    else:
        print("Having Null Data")

    print("==========加载DataLoader==========")
    dataloader_train = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True,
                                  num_workers=0, drop_last=False, collate_fn=collate_fn)
    dataloader_test = DataLoader(dataset_test, batch_size=config.batch_size, shuffle=False,
                                 num_workers=0, drop_last=True, collate_fn=collate_fn)
    first_data = next(iter(dataloader_train))
    # print(first_data)

    # 使用cuda:0
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device = config.device
    print('==> Loading model..')
    mdf_model = MDFTransformer()

    set_requires_grad(mdf_model,require_grad=True)
    num_update_steps_per_epoch = math.ceil(len(dataloader_train) / config.gradient_accumulation_steps)
    num_train_steps = num_update_steps_per_epoch * config.epochs
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(mdf_model.parameters(), lr=config.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=5, num_training_steps=num_train_steps
    )
    best_loss = np.inf
    for epoch in range(config.epochs):
        print(f"\n---------------------- Epoch: {epoch + 1} ---------------------------------- \n")
        ## Training Loop
        train_loss, train_report = TF.train_func_epoch(epoch + 1, mdf_model, dataloader_train, device,
                                                       optimizer,
                                                       scheduler, 0, 0)

        ## Validation loop
        val_loss, report, acc, prec, rec, f1_score = EF.eval_func(mdf_model, dataloader_test, device, criterion,epoch + 1)

        print(f"\nEpoch: {epoch + 1} | Training loss: {train_loss} | Validation Loss: {val_loss}")
        print()
        print("Train Report:")
        print(train_report)
        print()
        print("Validation Report:")
        print(report)
        print()
        print(f"Accuracy: {acc} | Micro Precision: {prec} | Micro Recall: {rec}, Micro F1-score: {f1_score} ")
        print(f"\n----------------------------------------------------------------------------")
