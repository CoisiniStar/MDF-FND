## Importing libraries
import os
import numpy as np
import pandas as pd

import dgl
from dgl.data import DGLDataset
import dgl.nn.pytorch as dglnn
from dgl.nn.pytorch import conv

import torch
from torch import nn
import torch.nn.functional as F
from utils import config
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = config.device

# 同构图嵌入层
class HomoEmbeddingLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 dropout_p=0.4):
        super(HomoEmbeddingLayer, self).__init__()

        ## Initialize the GAT Layers  初始化GAT层
        self.homographlayers = nn.ModuleDict({
            f"layer{i}": conv.GATConv(in_feats[i], out_feats[i],
                                      num_heads=1,
                                      feat_drop=0.4,
                                      attn_drop=0.0,
                                      residual=True, activation=nn.ELU(), bias=True)
            for i in range(len(in_feats)-1)
        }).to(device)

        self.dropout_p = dropout_p

    def forward(self, g):
        x_features = g.ndata['features'].to(device)

        ## Loop over the number of embedding layers
        for idx_layer, homographlayer in self.homographlayers.items():
            x_features = torch.mean(homographlayer(g, x_features), dim=1)

        return x_features


## Homo-Graph Classifier ##
class HomoGraphClassifier(nn.Module):
    def __init__(self,
                 in_feats_embedding=[768, 512],
                 out_feats_embedding=[512, 256],
                 classifier_dims=[256],
                 dropout_p=0.6,
                 n_classes=2):
        super(HomoGraphClassifier, self).__init__()

        ## GAT Layers initialized
        self.homoembeddinglayers = HomoEmbeddingLayer(in_feats=in_feats_embedding,
                                                      out_feats=out_feats_embedding,
                                                      dropout_p=dropout_p)
        ## Fake NEws Detector
        self.classifier = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 2)
        ).to(device)

    def forward(self, g):
        x_features = self.homoembeddinglayers(g)

        with g.local_scope():
            g.ndata['features'] = x_features

            # Calculate graph representation by average readout.
            hg = dgl.mean_nodes(g, 'features')

            return self.classifier(hg)