# 多模态

import numpy as np
import torch
import dgl
import torch.nn as nn
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
from utils import config
device = config.device
def Graph_DGL_Multi(text_embedding,img_embedding):
    image_vec = img_embedding
    image_adjMat = np.ones((img_embedding.shape[0], img_embedding.shape[0])).astype(float)
    image_pos_arr = np.where(image_adjMat > 0.5)
    graph_data = {
        ('image_node', 'image_edge', 'image_node'):
            (torch.tensor(image_pos_arr[0], dtype=torch.int64), torch.tensor(image_pos_arr[1], dtype=torch.int64))
    }
    text_vec = text_embedding
    text_adjMat = np.ones((text_embedding.shape[0], text_embedding.shape[0])).astype(float)
    text_pos_arr = np.where(text_adjMat > 0.5)
    graph_data[('text_node', 'text_edge', 'text_node')] = (torch.tensor(text_pos_arr[0], dtype=torch.int64),
                                                           torch.tensor(text_pos_arr[1], dtype=torch.int64))
    g = dgl.heterograph(graph_data)
    g = g.to(device)
    g.nodes["text_node"].data['features'] = text_embedding.clone().detach().requires_grad_(True)
    g.nodes["image_node"].data['features'] = img_embedding.clone().detach().requires_grad_(True)

    return g