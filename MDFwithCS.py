import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import bert_model
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings, BertModel, BertEncoder, BertLayer
from bert_model import BertCrossLayer, BertAttention

sys.path.append('../')
# from UEM.uem import UEM
import UEM.uem as uem
from Engine import init_weights
from utils import config
# from utils import config
device = config.device



class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        # self.clf.apply(init_weights.init_weights)

    def forward(self, x):
        x = self.clf(x)
        return x


class Classifier(nn.Module):
    def __init__(self, classifier_dims, classes):
        super(Classifier, self).__init__()
        layers = []
        # layers = torch.tensor(layers).to(device)
        num_layers = len(classifier_dims)
        # 构建前面的线性层
        for i in range(num_layers - 1):
            layers.append(nn.Linear(classifier_dims[i], classifier_dims[i + 1]).to(device))

            # 这里可以添加其他激活函数，如ReLU
            # layers.append(nn.ReLU())
        # 最后一层不加激活函数
        layers.append(nn.Linear(classifier_dims[-1], classes - 1))
        # 使用ModuleList来包装所有层
        self.layers = nn.ModuleList(layers)
        self.layers = self.layers.to(device)
    def forward(self, x):

        for layer in self.layers:
            x = layer(x)
        # 在最后一层后应用Sigmoid激活函数
        x = torch.sigmoid(x)
        return x

class Classifier2(nn.Module):
    def __init__(self, input_size, output_size):
        super(Classifier2, self).__init__()
        # 全连接层
        self.fc = nn.Linear(input_size, output_size)
        # Leaky ReLU 激活函数
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        # Dropout 层
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.fc(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        # 注意：因为最终的 CrossEntropyLoss 已经包含了 softmax，
        # 所以这里不需要再次应用 softmax。
        # 只有在需要直接获取概率输出时才应用 softmax。
        return x

class GatingMechanism(nn.Module):
    def __init__(self):
        super(GatingMechanism, self).__init__()

    def forward(self, x):
        return F.softmax(x, dim=0)


def DS_Combin(fea1, fea2, u1, u2):
    # 证据融合 --- 修改代码
    dims = 768
    # dims = [768,384,292,64,2]
    m1 = {'A': 0, 'B': 0, 'A∪B': u1}
    m2 = {'A': 0, 'B': 0, 'A∪B': u2}
    m = {'A': 0, 'B': 0, 'A∪B': 0}
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

    fea1 = fea1.to(device)
    fea2 = fea2.to(device)
    with torch.no_grad():
        # clasifier = Classifier(dims, 2)
        Classifier = Classifier2(dims,2).to(device)
        fea1o1 = Classifier(fea1).squeeze().to(device) # o1表示正类--真新闻
        fea2o1 = Classifier(fea2).squeeze().to(device)

    fea1o2 = 1 - fea1o1
    fea1o2 = fea1o2.to(device)
    fea2o2 = 1 - fea2o1
    fea2o2 = fea2o2.to(device)
    m1.update({"A": fea1o1})
    m1.update({"B": fea1o2})
    m2.update({"A": fea2o1})
    m2.update({"B": fea2o2})
    # 使用零填充的方式
    # padding = []
    # for key in m1.keys():
    #     tn1 = m1[key]
    #     tn2 = m2[key]
    #     shape1 = tn1.shape
    #     shape2 = tn2.shape
    #     for i in range(len(shape2)):
    #         padding.append((0,shape2[i]-shape1[i]))
    # print(padding.shape)

    padding_left = 0
    padding_right = m2['A'].shape[1] - m1['A'].shape[1]


    m1['A'] = F.pad(m1['A'],(padding_left,padding_right),'constant',1)
    m1['B'] = F.pad(m1['B'],(padding_left,padding_right),'constant',1)

    # 零填充的数量（在每个维度上填充的元素个数）
    # padding = (0, 48)  # 在第一个维度上不填充，在第二个维度上填充 48 个元素

    # 进行零填充
    # m1['A'] = torch.nn.functional.pad(m1['A'], padding)
    # m1['B'] = torch.nn.functional.pad(m1['B'], padding)

    m['A'] = m1['A'] * m2['A'] + m1['A'] * m2['A∪B'] + m1['A∪B'] * m2['A']
    m['B'] = m1['B'] * m2['B'] + m1['B'] * m2['A∪B'] + m1['A∪B'] * m2['B']
    m['A∪B'] = m1['A∪B'] * m2['A∪B']
    if m['A∪B'] > config.gammas:
        uncertainty = True
    else:
        uncertainty = False
    # 计算归一化因子K
    K = 1 - (m1['A'] * m2['B'] + m1['B'] * m2['A'])

    # 归一化组合后的置信度
    m_normalized = {k: v / K for k, v in m.items()}

    return m_normalized,uncertainty


# 使用门控网络与D-S理论结合
class GatedDSlayerWithFeatures(nn.Module):

    def __init__(self):
        super(GatedDSlayerWithFeatures, self).__init__()
        self.Gate = GatingMechanism()

    def forward(self, fea1, fea2, uncer1, uncer2):

        # 使用DS证据组合方式
        com_belief,uncertainty = DS_Combin(fea1, fea2, uncer1, uncer2)

        tensor1 = com_belief['A'].clone().detach()
        tensor1 = torch.unsqueeze(tensor1, 0).to(device)
        # tensor1 = torch.tensor(tensor1,requires_grad=False)
        # print(tensor1)
        tensor1 = tensor1.clone().detach().requires_grad_(False)
        # tensor2 = torch.tensor(com_belief['B']).to(device)
        tensor2 = com_belief['B'].clone().detach()
        tensor2 = torch.unsqueeze(tensor2, 0).to(device)
        tensor2 = tensor2.clone().detach().requires_grad_(False)

        tn = torch.cat((tensor1, tensor2), dim=0)
        tn = tn.clone().detach().requires_grad_(False)
        prob1, prob2 = self.Gate.forward(tn)
        if torch.gt(prob1[0, 0], prob2[0, 0]):
            predict = 1
        else:
            predict = 0
        return predict


class MDFTransformer(nn.Module):
    def __init__(self):
        super(MDFTransformer, self).__init__()
        self.gaussian = config.gaussian
        # self.Texviews = len(config.input_text_embedding)
        # self.Imgviews = len(config.input_image_embedding)

        bert_config = BertConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_layers,
            num_attention_heads=config.num_heads,
            intermediate_size=config.hidden_size * config.mlp_ratio,
            max_position_embeddings=config.max_text_len,
            hidden_dropout_prob=config.drop_rate,
            attention_probs_dropout_prob=config.drop_rate,
        )

        self.cross_modal_text_transform = nn.Linear(config.input_text_embed_size, config.hidden_size).to(device)
        self.cross_modal_text_transform.apply(init_weights.init_weights)
        self.cross_modal_image_transform = nn.Linear(config.input_image_embed_size, config.hidden_size).to(device)
        self.cross_modal_image_transform.apply(init_weights.init_weights)

        self.cross_modal_image_layers = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(config.num_top_layer)])
        self.cross_modal_image_layers.apply(init_weights.init_weights)
        self.cross_modal_text_layers = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(config.num_top_layer)]).to(device)
        self.cross_modal_text_layers.apply(init_weights.init_weights)
        self.Gatelayer = GatedDSlayerWithFeatures().to(device)

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
        device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

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
        split_txt = torch.split(text_embeds,1,dim=0)
        pred_list = []
        for item in zip(split_img,split_txt):
            img_embeds, txt_embeds, img_mu, img_logsigma, txt_mu, txt_logsigma = self.distr_modeling(item[0],
                                                                                                     item[1])
            # # 计算图像分布和N(ε,0,I)之间的KL散度
            # kl_img = -0.5 * torch.sum(1 + torch.log(img_logsigma) - img_mu ** 2 - img_logsigma)
            # kl_txt = -0.5 * torch.sum(1 + torch.log(txt_logsigma) - txt_mu ** 2 - txt_logsigma)  ###
            # # kl_img = torch.tensor(torch.mean(torch.sum(kl_img)),requires_grad=False)
            # # kl_txt = torch.tensor(torch.mean(torch.sum(kl_txt)),requires_grad=False)
            # kl_img = torch.mean(torch.sum(kl_img)).clone().detach().requires_grad_(True)
            # kl_txt = torch.mean(torch.sum(kl_txt)).clone().detach().requires_grad_(True)

            x, y = txt_embeds, img_embeds
            x = x.to(device)
            y = y.to(device)
            # diff = y.size(0) - x.size(0)
            # x = F.pad(x,(0,0,0,0,0,diff))
            # 交叉注意力
            for text_layer, img_layer in zip(self.cross_modal_text_layers, self.cross_modal_image_layers):
                # torch.cuda.empty_cache()
                text_layer = text_layer.to(device)
                img_layer = img_layer.to(device)
                x1 = text_layer(x, y, None, None)
                y1 = img_layer(y, x, None, None) # CUDA out of memory.
                x, y = x1[0], y1[0]  # x是文本层，y是图像层
                x = x.to(device)
                y = y.to(device)
            # 计算不确定性得分
            img_score, txt_score = self.pgu(img_mu, img_logsigma, txt_mu, txt_logsigma)

            # GDF
            pred = self.Gatelayer(y, x, img_score, txt_score)
            pred_list.append(pred)
            count_0 = pred_list.count(0)
            count_1 = pred_list.count(1)
            # print('count_0:',count_0)
            # print('count_1:',count_1)
        return pred_list

