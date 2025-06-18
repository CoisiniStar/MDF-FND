
import json
import numpy as np
import os
from PIL import Image

import torch
from torch.utils.data import Dataset
from torch import  nn
import pandas as pd

import utils.config as config
# import config
class UEMDataset(Dataset):
    def __init__(self,df,root_dir,image_id,text_id,image_vec_dir,text_vec_dir):
        # super(UNDataset, self).__init__()
        self.df = df
        self.root_dir = root_dir
        self.image_id = image_id
        self.text_id = text_id
        self.image_vec_dir = image_vec_dir
        self.text_vec_dir = text_vec_dir
        self.adaptive_pooling = nn.AdaptiveAvgPool1d(768)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # filenames for the idx
        file_name_image = self.df[self.image_id][idx].split(",")[0]
        file_name_text = self.df[self.text_id][idx]

        file_name = f"{self.root_dir}{self.image_vec_dir}{file_name_image}_full_image.npy"
        new_file_path = ''.join(file_name.split())

        image_vec_full = np.load(new_file_path)
        ## Load node embeddings for objects present in the text
        try:
            image_vec = np.load(f'{self.root_dir}{self.image_vec_dir}{file_name_image}.npy')
            all_image_vec = np.concatenate([image_vec_full, image_vec], axis=0)
            image_vec = all_image_vec
        except:
            image_vec = image_vec_full

        ## Resize the image vectors to match the text embedding dimension
        image_vec = self.adaptive_pooling(torch.tensor(image_vec).float().unsqueeze(0)).squeeze(0)
        ## Load node embeddings for tokens present in the text
        text_vec = np.load(f'{self.root_dir}{self.text_vec_dir}{file_name_text}.npy')

        ## Load full image node embedding
        text_vec_full = np.load(f'{self.root_dir}{self.text_vec_dir}{file_name_text}_full_text.npy')

        all_text_vec = np.concatenate([text_vec_full, text_vec], axis=0)
        text_vec = all_text_vec
        text_vec = torch.from_numpy(text_vec)
        # print(type(text_vec))
        ## Node embeddings for the multimodal graph
        all_vec = np.concatenate([text_vec, image_vec], axis=0)

        ## find the label
        if self.df['label'][idx] == 'real':
            label = 0
        elif self.df['label'][idx] == 'fake':
            label = 1
        return image_vec,text_vec,label


# df_train = pd.read_csv(f'{config.root_dir}{config.me15_train_csv_name}',encoding='utf-8')
# df_train = df_train.dropna().reset_index(drop=True)
# dataset_train = UEMDataset(df_train, config.root_dir, "imageId(s)", "tweetId",
#                                          config.me15_image_vec_dir, config.me15_text_vec_dir)
# print(dataset_train[0])


# df_train = pd.read_csv(f'{config.root_dir}{config.me15_train_csv_name}',encoding='utf-8')
# df_train = df_train.dropna().reset_index(drop=True)
# dataset_train = UEMDataset(df_train, config.root_dir, "imageId(s)", "tweetId",
#                                          config.me15_image_vec_dir, config.me15_text_vec_dir)
# print(dataset_train[0])



























import json
import numpy as np
import os
from PIL import Image

import torch
from torch.utils.data import Dataset
from torch import  nn
import pandas as pd

import utils.config as config
# import config
class UEMDataset(Dataset):
    def __init__(self,df,root_dir,image_id,text_id,image_vec_dir,text_vec_dir):
        # super(UNDataset, self).__init__()
        self.df = df
        self.root_dir = root_dir
        self.image_id = image_id
        self.text_id = text_id
        self.image_vec_dir = image_vec_dir
        self.text_vec_dir = text_vec_dir
        self.adaptive_pooling = nn.AdaptiveAvgPool1d(768)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # filenames for the idx
        file_name_image = self.df[self.image_id][idx].split(",")[0]
        file_name_text = self.df[self.text_id][idx]

        file_name = f"{self.root_dir}{self.image_vec_dir}{file_name_image}_full_image.npy"
        new_file_path = ''.join(file_name.split())

        image_vec_full = np.load(new_file_path)
        ## Load node embeddings for objects present in the text
        try:
            image_vec = np.load(f'{self.root_dir}{self.image_vec_dir}{file_name_image}.npy')
            all_image_vec = np.concatenate([image_vec_full, image_vec], axis=0)
            image_vec = all_image_vec
        except:
            image_vec = image_vec_full

        ## Resize the image vectors to match the text embedding dimension
        image_vec = self.adaptive_pooling(torch.tensor(image_vec).float().unsqueeze(0)).squeeze(0)
        ## Load node embeddings for tokens present in the text
        text_vec = np.load(f'{self.root_dir}{self.text_vec_dir}{file_name_text}.npy')

        ## Load full image node embedding
        text_vec_full = np.load(f'{self.root_dir}{self.text_vec_dir}{file_name_text}_full_text.npy')

        all_text_vec = np.concatenate([text_vec_full, text_vec], axis=0)
        text_vec = all_text_vec
        text_vec = torch.from_numpy(text_vec)
        # print(type(text_vec))
        ## Node embeddings for the multimodal graph
        all_vec = np.concatenate([text_vec, image_vec], axis=0)

        ## find the label
        if self.df['label'][idx] == 'real':
            label = 0
        elif self.df['label'][idx] == 'fake':
            label = 1
        return image_vec,text_vec,label


# df_train = pd.read_csv(f'{config.root_dir}{config.me15_train_csv_name}',encoding='utf-8')
# df_train = df_train.dropna().reset_index(drop=True)
# dataset_train = UEMDataset(df_train, config.root_dir, "imageId(s)", "tweetId",
#                                          config.me15_image_vec_dir, config.me15_text_vec_dir)
# print(dataset_train[0])
