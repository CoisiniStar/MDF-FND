import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from thop import profile
sys.path.append('../')
import UEM.uem as uem
from Engine import init_weights
from utils import config
# from DFN-Turbo import GATlayer
from GATLayer_unimodal import HomoGraphClassifier
from GATLayer_multimodal import HeteroGraphClassifier
import CGM,CUG
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = config.device


"""加入多元高斯分布的熵损失函数"""
def margin_entropy_loss(margin, logsigma):
    feat_dim = logsigma.shape[-1]
    entropy = float(feat_dim / 2 * (np.log(2 * np.pi) + 1)) + torch.sum(logsigma, -1) / 2
    zero = torch.zeros_like(entropy)
    loss = torch.max(margin - entropy, zero)
    loss = torch.mean(loss)
    return loss

def DS_Combin(fea1o,fea2o,fea1score,fea2score):  # 需要传递归一化后的决策值;每个模态的置信度
    # 证据融合 --- 修改代码
    m1 = {'A': 0, 'B': 0, 'A∪B': fea1score}
    m2 = {'A': 0, 'B': 0, 'A∪B': fea2score}
    m = {'A': 0, 'B': 0, 'A∪B': 0}
    fea1o = fea1o.squeeze(0).squeeze(0)
    fea2o = fea2o.squeeze(0).squeeze(0)
    fea1o1 = fea1o[0]
    fea1o2 = fea1o[1]
    fea2o1 = fea2o[0]
    fea2o2 = fea2o[1]
    m1.update({"A": fea1o1})
    m1.update({"B": fea1o2})
    m2.update({"A": fea2o1})
    m2.update({"B": fea2o2})

    m['A'] = m1['A'] * m2['A'] + m1['A'] * m2['A∪B'] + m1['A∪B'] * m2['A']
    m['B'] = m1['B'] * m2['B'] + m1['B'] * m2['A∪B'] + m1['A∪B'] * m2['B']
    m['A∪B'] = m1['A∪B'] * m2['A∪B']
    # print('confidence:'+m['A∪B'].__str__())

    if m['A∪B'] > config.gamma:  # 0.5   0.35
        uncertainty = True
    else:
        uncertainty = False
    # print('uncertainty:',uncertainty)
    # 计算归一化因子K
    K = 1 - (m1['A'] * m2['B'] + m1['B'] * m2['A'])

    # 归一化组合后的置信度
    m_normalized = {k: v / K for k, v in m.items()}
    if (fea1score > fea2score):
        # 用fea1单模态决策
        predict_index = 0      # 第一个特征:文本
    else:
        # 用fea2单模态决策
        predict_index = 1       # 第二个特征:图像
    return predict_index,uncertainty


class MDFTransformer(nn.Module):
    def __init__(self):
        super(MDFTransformer, self).__init__()
        self.gaussian = config.gaussian
        self.txtgraphclassifier = HomoGraphClassifier(
                 # in_feats_embedding=[6,89,768],
                 in_feats_embedding=[768, 512],
                 out_feats_embedding=[768, 256],
                 classifier_dims=[256],
                 dropout_p=0.4,
                 n_classes=2)
        self.imggraphclassifier = HomoGraphClassifier(
            # in_feats_embedding=[6, 32, 768],
            in_feats_embedding=[768, 512],
            out_feats_embedding=[768, 256],
            classifier_dims=[256],
            dropout_p=0.4,
            n_classes=2)
        self.Heterographclassifier = HeteroGraphClassifier(
                 in_feats_embedding= [768, 512],
                 out_feats_embedding= [512, 256],
                 classifier_dims=[128],
                 dropout_p=0.6,
                 n_classes=2)
        if self.gaussian:
            # 将PDE的隐藏层大小设置为768 同时使用init_weights参数
            self.img_gau_encoder = uem.DisTrans(768, 12).to(device)
            self.txt_gau_encoder = uem.DisTrans(768, 12).to(device)
            self.img_gau_encoder.apply(init_weights.init_weights)
            self.txt_gau_encoder.apply(init_weights.init_weights)
            # 采样大小
            self.sample_num = config.sample_num
            # mu_num 阈值
            self.mu_num = config.mu_num

    def distr_modeling(self, image_embeds,
                       text_embeds):
        device = config.device
        image_embeds = image_embeds.to(device)
        text_embeds = text_embeds.to(device)
        img_mu, img_logsigma, _ = self.img_gau_encoder(image_embeds, mask=None)

        # if self.training:  # 训练阶段打印图像的logsigma
        # print('img_sigma_mean', torch.mean(torch.exp(img_logsigma)))
        z = [img_mu] * self.mu_num
        for i in range(self.sample_num):
            eps = torch.randn(img_mu.shape[0], img_mu.shape[1], img_mu.shape[2]).to(device)
            z1 = img_mu + torch.exp(img_logsigma) * eps
            z.append((z1))
        image_embeds = torch.cat(z)
        txt_mu, txt_logsigma, _ = self.txt_gau_encoder(text_embeds, mask=None)
        # if self.training:
        #     print('text_sigma_mean', torch.mean(torch.exp(txt_logsigma)))
        z = [txt_mu] * self.mu_num
        for i in range(self.sample_num):
            eps = torch.randn(txt_mu.shape[0], txt_mu.shape[1], txt_mu.shape[2]).to(device)
            z1 = txt_mu + torch.exp(txt_logsigma) * eps
            z.append(z1)
        text_embeds = torch.cat(z)
        return image_embeds, text_embeds, img_mu, img_logsigma, txt_mu, txt_logsigma

    # 重塑
    def pgu(self, img_mu, img_logsigma, txt_mu, txt_logsigma, scaling_factor=1.0):
        # 利用每个模态的均值和方差计算每个模态的不确定性分数
        img_logsigma = torch.max(img_logsigma)
        txt_logsigma = torch.max(txt_logsigma)
        img_mu = torch.max(img_mu)
        txt_mu = torch.max(txt_mu)

        std_img = math.sqrt(img_logsigma)
        std_txt = math.sqrt(txt_logsigma)

        uncertaintyScoe_img = (std_img / img_mu) * scaling_factor
        uncertaintyScore_txt = (std_txt / txt_mu) * scaling_factor
        img_score = torch.sigmoid(uncertaintyScoe_img)

        txt_score = torch.sigmoid(uncertaintyScore_txt)
        return img_score, txt_score

    def forward(self, img_embeds, text_embeds):
        # UEM  (batch_size,sentence/img,768bert)
        # 后面的四个参数用来定义损失函数
        split_img = torch.split(img_embeds, 1, dim=0)
        split_txt = torch.split(text_embeds, 1, dim=0)
        pred_list = []

        for item in zip(split_img, split_txt):
            img_embeds, txt_embeds, img_mu, img_logsigma, txt_mu, txt_logsigma = self.distr_modeling(item[0],
                                                                                                     item[1])

            # 利用pgu计算不确定性得分
            img_score,txt_score = self.pgu(img_mu,img_logsigma,txt_mu,txt_logsigma)
            x, y = txt_embeds, img_embeds
            ### 下面使用GAT进行Attention融合
            margin_loss1 = margin_entropy_loss(config.margin_value,img_logsigma)
            margin_loss2 = margin_entropy_loss(config.margin_value,txt_logsigma)
            margin_loss = (margin_loss1+margin_loss2)/2
            ## 建立图结构
            # 单模态文本图结构
            g_t = CUG.Graph_DGL_TM(x).to(device)
            # 单模态图像图结构
            g_e = CUG.Graph_DGL_TM(y).to(device)
            # 多模态图结构
            g_m = CGM.Graph_DGL_Multi(x,y).to(device)
            deci_t = self.txtgraphclassifier.forward(g_t)
            deci_t = deci_t.squeeze(0)
            # print('deci_t shape:',deci_t.shape)
            deci_e = self.imggraphclassifier.forward(g_e)
            deci_e = deci_e.squeeze(0)
            # print('deci_e shape:',deci_e.shape)

            deci_m = self.Heterographclassifier.forward(g_m)
            """deci_m shape: torch.Size([1, 2, 1, 2])"""
            # deci_m = deci_m.squeeze(0)
            """deci_m shape: torch.Size([2, 1, 2])"""
            # deci_m = deci_m.unsqueeze(0)
            # deci_m = deci_m.view(-1,2)
            deci_m = deci_m.flatten()
            deci_m = deci_m.reshape(-1,2)
            max_values = torch.max(deci_m,dim=0)[0]

            # print('max_values:',deci_m)
            # max_values = torch.tensor(torch.max(deci_m,dim=0))
            deci_m = torch.unsqueeze(max_values,dim=0)

            # print('deci_m shape:', deci_m.shape)
            """
                deci_t shape: torch.Size([1, 2])
                deci_e shape: torch.Size([1, 2])
                deci_m shape: torch.Size([2, 2])
            """
            fea1o = torch.sigmoid(deci_t)
            fea2o = torch.sigmoid(deci_e)
            predict,uncertain = DS_Combin(fea1o,fea2o,txt_score,img_score)
            # pred_list.append(deci_m)
            if uncertain==True:
                # return deci_m
                pred_list.append(deci_m)      # Tensor([1,1,2])
            elif uncertain==False and predict==0:
                pred_list.append(deci_t)
            else:
                pred_list.append(deci_e)
        return pred_list,margin_loss


# from fvcore.nn import FlopCountAnalysis, parameter_count_table
# model = MDFTransformer()
# input1 = torch.rand(16,20,768)
# input2 = torch.rand(16,30,768)
# flops = FlopCountAnalysis(model,(input1,input2))
# print("FLOPs: ", flops.total())
# def print_model_parm_nums(model):
#     total = sum([param.nelement() for param in model.parameters()])
#     print('  + Number of params: %.2fM' % (total / 1e6))
#
# print_model_parm_nums(model)
