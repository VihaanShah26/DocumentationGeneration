from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.optim as optim

import torch
import json
import numpy as np
from tqdm import tqdm


class DocDataset(torch.utils.data.Dataset):

    def __init__(self, encodings, tokenizer):
        self.encodings = encodings
        self.tokenizer = tokenizer
        

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, index):
        encoding = self.encodings[index]
        tokenized = self.tokenizer(encoding, return_tensors='pt', padding='max_length', truncation=True, max_length=512)

        input_ids = tokenized.input_ids.squeeze()
        attention_mask = tokenized.attention_mask.squeeze()

        doc_token_id = self.tokenizer.encode("[DOC]")[0]
        doc_start_pos = (input_ids == doc_token_id).nonzero(as_tuple=True)[0].item()

        labels = torch.full_like(input_ids, -100)
        labels[doc_start_pos + 1:] = input_ids[doc_start_pos + 1:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

        
def main():
    special_tokens_dict = {"additional_special_tokens": ["[START]", "[DOC]"]}
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", padding_side='left')
    tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    optimizer = optim.Adam(model.parameters(), lr=3e-5)

    train, valid, test = [], [], []

    with open('ProcessedTrainData.json') as json_file:
        json_list = list(json_file)
    
    for i in range(len(json_list)):
        json_str = json_list[i]
        result = json.loads(json_str)

        tokenizer.add_tokens(result['code_tokens'])
        tokenizer.add_tokens(result['docstring_tokens'])

        encoding = "[START] " + result['code'].replace("\n", " ") + " [DOC] " + result['docstring'].replace("\n", " ")
        encoding = " ".join(encoding.split())

        train.append(encoding)
    
    with open('ProcessedValidationData.json') as json_file:
        json_list = list(json_file)
    for i in range(len(json_list)):
        json_str = json_list[i]
        result = json.loads(json_str)

        tokenizer.add_tokens(result['code_tokens'])
        tokenizer.add_tokens(result['docstring_tokens'])

        encoding = "[START] " + result['code'].replace("\n", " ") + " [DOC] " + result['docstring'].replace("\n", " ")
        encoding = " ".join(encoding.split())

        valid.append(encoding)

    with open('ProcessedTestData.json') as json_file:
        json_list = list(json_file)
    
    for i in range(len(json_list)):
        json_str = json_list[i]
        result = json.loads(json_str)

        tokenizer.add_tokens(result['code_tokens'])
        tokenizer.add_tokens(result['docstring_tokens'])

        encoding = "[START] " + result['code'].replace("\n", " ") + " [DOC] " + result['docstring'].replace("\n", " ")
        encoding = " ".join(encoding.split())

        test.append(encoding)


    train_dataset = DocDataset(train, tokenizer)
    print(train_dataset[0])

    

if __name__ == "__main__":
    main()