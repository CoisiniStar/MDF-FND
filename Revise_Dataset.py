import torch
import pandas as pd
import numpy as np
import transformers
import torchvision
from torchvision import transforms
from PIL import Image
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import glob
import torch.nn.functional as F
from transformers import BertModel
import random
import time

import re

label2id = {"real":0, "fake":1}
"""找到对应的图片"""
import os
import glob


def find_image_files(folder_path, image_name_without_extension):
    # Define possible image extensions
    possible_extensions = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff']
    # Iterate over each extension to find the file
    for ext in possible_extensions:
        # Create a search pattern
        pattern = os.path.join(folder_path, f"{image_name_without_extension}.{ext}").strip().replace(" ", "")
        # Use glob to find files matching the pattern
        found_files = glob.glob(pattern)
        # If any files are found, return the first one
        if found_files:
            return found_files[0]
    # If no files are found, return None or raise an error
    return None


def open_images(image_files):
    # images = []
    for image_file in image_files:
        try:
            img = Image.open(image_file).convert("RGB")
            # images.append(img)
        except Exception as e:
            print(f"Failed to open {image_file}: {e}")
    return img


def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


class FakeNewsDataset(Dataset):
    """Fake News Dataset"""

    def __init__(self, df, root_dir, image_transform, tokenizer, MAX_LEN):
        """
        Args:
            csv_file (string): Path to the csv file with text and img name.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = False
        self.csv_data = df
        self.root_dir = root_dir
        self.image_transform = image_transform
        self.tokenizer_bert = tokenizer
        self.MAX_LEN = MAX_LEN
        self.image_dir = "../Image_Temp/"
        self.image_name = "imageId(s)"
    def __len__(self):
        return self.csv_data.shape[0]

    def pre_processing_BERT(self, sent):
        # Create empty lists to store outputs
        input_ids = []
        attention_mask = []

        encoded_sent = self.tokenizer_bert.encode_plus(
            text=text_preprocessing(sent),  # Preprocess sentence
            add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
            max_length=self.MAX_LEN,  # Max length to truncate/pad
            padding='max_length',  # Pad sentence to max length
            return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True,  # Return attention mask
            truncation=True
        )

        input_ids = encoded_sent.get('input_ids')
        attention_mask = encoded_sent.get('attention_mask')

        # Convert lists to tensors
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)

        return input_ids, attention_mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # image_name_without_extension = self.csv_data[self.image_name][idx].split(",")[0]
        image_name_without_extension = self.csv_data[self.image_name][idx].split(",")[0].strip().replace(" ", "")
        #         image = io.imread(img_name)
        image_files = find_image_files(self.image_dir, image_name_without_extension)
        # image_files = image_files
        # images = open_images(image_files)
        if image_files is not None:
            images = Image.open(image_files).convert("RGB")
            # print('length of images:\n',len(images))
            if self.transform:
                images = self.image_transform(images)
        else:
            print(f'Image:{image_name_without_extension} is not found\n')
            raise FileExistsError
        # print("images:",images)

        # images = self.image_transform(images)
        text = self.csv_data['tweetText'][idx]

        tensor_input_id, tensor_input_mask = self.pre_processing_BERT(text)

        label = self.csv_data['label'][idx]
        label = label2id.get(label, -1)
        label = torch.tensor(label)
        """报错：IndexError: list index out of range"""
        # images = images[0]
        sample = {'image': images, 'BERT_ip': [tensor_input_id, tensor_input_mask], 'label': label}
        # print(idx)

        return sample