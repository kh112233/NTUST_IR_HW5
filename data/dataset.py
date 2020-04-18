import pandas as pd
import numpy as np

import torch
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
        if self.train:
            return len(self.raw_data)
        #else:
        #    doc_size = len(self.raw_data['381'])
        #    return int(len(self.raw_data)*doc_size)

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
            doc_size = len(self.raw_data['381'])
            query_name = str(idx//doc_size + 381)
            doc_name = self.raw_data[query_name][idx%doc_size]

            with open(f'{self.test_dir}/query/{query_name}') as q:
                query = q.read()
            with open(f'{self.doc_dir}/{doc_name}') as d:
                doc = d.read()

            ids, token_type, attention_mask = self._preprocess(query, doc)
            ids = torch.LongTensor(ids)
            token_type = torch.LongTensor(token_type)
            attention_mask = torch.LongTensor(attention_mask)
            return ids, token_type, attention_mask

def train_collate_fn(batch):
    ids, token_type, attention_mask, labels = zip(*batch)
    ids = nn.utils.rnn.pad_sequence(ids, batch_first=True)
    token_type = nn.utils.rnn.pad_sequence(token_type, batch_first=True)
    attention_mask = nn.utils.rnn.pad_sequence(attention_mask, batch_first=True)
    labels = torch.cat(labels)
    return ids, token_type, attention_mask, labels

def test_collate_fn(batch):
    ids, token_type, attention_mask = zip(*batch)
    ids = nn.utils.rnn.pad_sequence(ids, batch_first=True)
    token_type = nn.utils.rnn.pad_sequence(token_type, batch_first=True)
    attention_mask = nn.utils.rnn.pad_sequence(attention_mask, batch_first=True)
    return ids, token_type, attention_mask   

def train_valid_split(train_dir, valid_ratio):
    # Get training data
    pos_data = pd.read_csv(train_dir + 'Pos.txt', sep=" ", names=['querys', 'documents', 'labels'])
    neg_data = pd.read_csv(train_dir + 'Neg.txt', sep=" ", names=['querys', 'documents', 'labels'])

    # split train/valid data
    raw_train_data = np.concatenate([pos_data, neg_data])
    np.random.shuffle(raw_train_data)
    valid_data = raw_train_data[:int(len(raw_train_data)*valid_ratio)]
    train_data = raw_train_data[int(len(raw_train_data)*valid_ratio):]

    """
    train_dataset = QueryDocumentDataset(train_dir=train_dir, test_dir=test_dir, doc_dir=doc_dir, raw_data=train_data)
    valid_dataset = QueryDocumentDataset(train_dir=train_dir, test_dir=test_dir, doc_dir=doc_dir, raw_data=val_data)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=train_collate_fn)
    val_dataloader = DataLoader(valid_dataset, batch_size=batch_size*8, shuffle=True, num_workers=4, collate_fn=train_collate_fn)

    # Get testing data
    doc_list = [doc for doc in listdir(doc_dir) if isfile(join(doc_dir, doc))]
    with open(test_dir+"/query_list.txt") as f:
        query_list = f.read().split()
    raw_test_dict = {}
    for query in query_list:
        raw_test_dict[query] = doc_list
    
    test_dataset = QueryDocumentDataset(train_dir=train_dir, test_dir=test_dir, doc_dir=doc_dir, raw_data=raw_test_dict, train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=4, collate_fn=test_collate_fn)    
    """

    return train_data, valid_data