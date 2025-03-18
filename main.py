from transformers import AutoTokenizer, BertModel, GPT2LMHeadModel, GPT2Tokenizer
import torch.optim as optim

import torch
import json
import numpy as np
from tqdm import tqdm


class DocDataset(torch.utils.data.Dataset):

    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass


def load_data():
    with open('ProcessedTrainData.json') as json_file:
        json_list = list(json_file)
        print(json_list[0])


if __name__ == "__main__":
    load_data()