import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset

import re
from transformers import*

class QueryDocumentDataset(Dataset):
    
    def __init__(self, query_dir, doc_dir, df, train=True):
        """
        :param query_dir:   Directory path of query files.
        :param doc_dir:     Directory path of document files.
        :param df:          Dataframe of input.      
        :param train:       The data if for training or testing.
        """

        # Setting directory path
        self.query_dir = query_dir
        self.doc_dir = doc_dir

        # Setting raw data of dataset and tokenizer of BERT
        self.df = df
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Setting flag for training or testing dataset
        self.train = train
    
    def _preprocess(self, query, doc):
        # Query   
        cls_token = self.tokenizer.cls_token
        sep_token = self.tokenizer.sep_token
        query = re.sub(r'[^\w]', ' ', query)
        query = query.split()
        query = f"{cls_token} {query} {sep_token}"
        query_tokens = self.tokenizer.tokenize(query)
        query_ids = self.tokenizer.convert_tokens_to_ids(query_tokens)
        cls_ids = self.tokenizer.convert_tokens_to_ids(cls_token)
        sep_ids = self.tokenizer.convert_tokens_to_ids(sep_token)
        query_ids = [cls_ids] + query_ids + [sep_ids]

        # Document
        doc_len = 512 - len(query_ids)
        doc = re.sub(r'[^\w]', ' ', doc)
        doc = doc.split()
        doc = f"{' '.join(doc)}"
        doc_tokens = self.tokenizer.tokenize(doc)
        doc_ids = self.tokenizer.convert_tokens_to_ids(doc_tokens)
        doc_ids = doc_ids[:doc_len]
        
        ids = query_ids + doc_ids      
        
        token_type = [*(0 for _ in query_ids), *(1 for _ in doc_ids)]
        assert len(token_type) == len(ids)

        attention_mask = [*(1 for _ in query_ids), *(1 for _ in doc_ids)]
        assert len(attention_mask) == len(ids)

        return ids, token_type, attention_mask
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        if self.train:
            with open(f'{self.query_dir}/query/{self.df[idx][0]}') as q:
                query = q.read()
            with open(f'{self.doc_dir}/{self.df[idx][1]}') as d:
                doc = d.read()
            
            ids, token_type, attention_mask = self._preprocess(query, doc)
            label = self.df[idx][2]

            ids = torch.LongTensor(ids)
            token_type = torch.LongTensor(token_type)
            attention_mask = torch.LongTensor(attention_mask)
            label = torch.FloatTensor([label])
            return ids, token_type, attention_mask, label

        else:
            with open(f'{self.query_dir}/query/{self.df[idx][0]}') as q:
                query = q.read()
            with open(f'{self.doc_dir}/{self.df[idx][1]}') as d:
                doc = d.read()

            ids, token_type, attention_mask = self._preprocess(query, doc)
            ids = torch.LongTensor(ids)
            token_type = torch.LongTensor(token_type)
            attention_mask = torch.LongTensor(attention_mask)
            return ids, token_type, attention_mask, self.df[idx][0], self.df[idx][1]

def train_collate_fn(batch):
    ids, token_type, attention_mask, labels = zip(*batch)
    ids = nn.utils.rnn.pad_sequence(ids, batch_first=True)
    token_type = nn.utils.rnn.pad_sequence(token_type, batch_first=True)
    attention_mask = nn.utils.rnn.pad_sequence(attention_mask, batch_first=True)
    labels = torch.cat(labels)
    return ids, token_type, attention_mask, labels

def test_collate_fn(batch):
    ids, token_type, attention_mask, qname, dname = zip(*batch)
    ids = nn.utils.rnn.pad_sequence(ids, batch_first=True)
    token_type = nn.utils.rnn.pad_sequence(token_type, batch_first=True)
    attention_mask = nn.utils.rnn.pad_sequence(attention_mask, batch_first=True)
    return ids, token_type, attention_mask , qname, dname

def train_valid_split(train_dir, valid_ratio):
    # Get training data
    pos_data = pd.read_csv(train_dir + 'Pos.txt', sep=" ", names=['querys', 'documents', 'labels'])
    neg_data = pd.read_csv(train_dir + 'Neg.txt', sep=" ", names=['querys', 'documents', 'labels'])

    # split train/valid data
    raw_train_data = np.concatenate([pos_data, neg_data])
    np.random.shuffle(raw_train_data)
    valid_data = raw_train_data[:int(len(raw_train_data)*valid_ratio)]
    train_data = raw_train_data[int(len(raw_train_data)*valid_ratio):]

    return train_data, valid_data