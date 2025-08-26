
# 单模态--文本
import numpy as np
import torch
import dgl
from utils import config

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = config.device

def Graph_DGL_TM(embeddings):
    text_adjMat = np.ones((embeddings.shape[0], embeddings.shape[0])).astype(float)
    text_pos_arr = np.where(text_adjMat > 0.5)
    edges = (torch.tensor(np.concatenate((text_pos_arr[0], text_pos_arr[1])), dtype=torch.int64),
                   torch.tensor(np.concatenate((text_pos_arr[1], text_pos_arr[0])), dtype=torch.int64))
    g = dgl.graph(edges)
    g = g.to(device)
    # g.ndata['features'] = torch.tensor(embeddings).float()
    g.ndata['features'] = embeddings.clone().detach().requires_grad_(True)
    return g