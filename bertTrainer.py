# Import

import time
import copy
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
import random
import statistics

import re

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from transformers import*

# Set Seed

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

class QueryDocumentDataset(Dataset):
    
    def __init__(self, train_dir, test_dir, doc_dir, raw_data, train=True):
        """
        :param train_dir: training data directory path
        :param test_dir: tresting data directory path
        :param doc_dir: document data directory path
        :param raw_data: every query with all documents
        :param train: True if raw_data is training data
        """

        # Setting directory path
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.doc_dir = doc_dir

        # Setting raw data of dataset and tokenizer of BERT
        self.raw_data = raw_data
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
        else:
            doc_size = len(self.raw_data['381'])
            return int(len(self.raw_data)*doc_size)

    def __getitem__(self, idx):

        if self.train:
            with open(f'{self.train_dir}/query/{self.raw_data[idx][0]}') as q:
                query = q.read()
            with open(f'{self.doc_dir}/{self.raw_data[idx][1]}') as d:
                doc = d.read()
            
            ids, token_type, attention_mask = self._preprocess(query, doc)
            label = self.raw_data[idx][2]

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


class BERTTrainer:
    
    def __init__(self, bert_finetune:BertForSequenceClassification,
                 train_dataloader:DataLoader, val_dataloader=None, test_dataloader=None, test_dict=None,
                 batch_size = 2, accm_batch_size = 32, lr = 2e-5,
                 with_cuda: bool = True):
        """
        :param bert_finetune: BERT finetune model
        :param train_dataloader: train dataset data loader
        :param val_dataloader: valid dataset data loader (can be None)
        :param test_dataloader: test dataset data loader (can be None)
        :param batch_size: batch size of data loader 
        :param accm_batch_size: accumulate gradient batch size
        :param lr: learning rate of AdamW
        :param with_cuda: training with cuda
        """
        
        # Setup cuda device for BERT model training
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        # Get BERT finetune model
        self.bert_finetune = bert_finetune
        self.bert_finetune.to(self.device)

        # Setting the AdamW optimizer with hyper-parameter
        self.optimizer = optim.AdamW(self.bert_finetune.parameters(), lr=lr) # suggest lr: 2e-5~3e-5

        # Setting train and test data loader
        self.train_data = train_dataloader
        self.val_data = val_dataloader
        self.test_data = test_dataloader
        self.test_dict = test_dict

        # Using binary cross entropy with logists loss for criterion
        self.criterion = nn.BCEWithLogitsLoss()

        # Setting best BCE loss and accuracy for early stopping
        self.best_BCE = 1000
        self.best_acc = 0

        # Record best weight
        self.best_BCE_weight = None
        self.best_acc_weight = None

        # Setting accmulate gradient
        self.accm_batch_size = accm_batch_size
        self.batch_size = batch_size
        assert self.accm_batch_size % self.batch_size == 0
        self.accm_steps = self.accm_batch_size // self.batch_size

        # Setting patience of early stop
        self.patience = 0
    
    def set_test_dataloader(self, data_loader, test_dict):
        self.test_data = data_loader
        self.test_dict = test_dict

    def train(self, epochs=1000, early_stop=False, patience=None):
        self.iteration(self.train_data, epochs, early_stop, patience, split="train")
    
    def test(self, load_weights=False):
        if load_weights != False:
            print('Loading pretrained weights...')
            self.bert_finetune.load_state_dict(torch.load(load_weights))
        else:
            self.bert_finetune.load_state_dict(self.best_BCE_weight)
        self.iteration(self.test_data, split="test")
    
    def _early_stop(self, val_BCE, val_acc):
        if val_BCE < self.best_BCE:
            self.best_BCE = val_BCE
            self.best_BCE_weight = copy.deepcopy(self.bert_finetune.state_dict()) 
            torch.save(self.best_BCE_weight,
                        f"./weights/first500/loss/" + str(self.best_BCE)[:8] + ".ckpt") #看你要存到哪裡
            self.patience = 0
            
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.best_acc_weight = copy.deepcopy(self.bert_finetune.state_dict()) 
            torch.save(self.best_acc_weight,
                        f"./weights/first500/accuracy/" + str(self.best_acc)[:8] + ".ckpt") #看你要存到哪裡
            self.patience = 0
        
        else:
            self.patience+=1

    def _val(self, dataloader):
        self.bert_finetune.eval() 
        with torch.no_grad():
            sigmoid = nn.Sigmoid()
            y_logits, y_true = np.zeros(0), np.zeros(0)

            for idx, batch in enumerate(dataloader):
                batch = [*(tensor.cuda() for tensor in batch)]
                ids, token_type, attention_mask, labels = batch

                loss, logits = self.bert_finetune(input_ids=ids, token_type_ids=token_type, attention_mask=attention_mask, labels=labels)
                y_true = np.concatenate([y_true, labels.cpu().numpy()])
                y_logits = np.concatenate([y_logits, logits.cpu().numpy().reshape(logits.shape[0])])

            y_scores = sigmoid(torch.Tensor(y_logits))
            loss = self.criterion(torch.Tensor(y_logits).squeeze(-1), torch.Tensor(y_true))
            loss = loss.cpu().item()
            accuracy = (y_scores.cpu().numpy().round()==y_true).mean()

        return loss, accuracy

    def save(self, doc_scores):
        doc_scores = doc_scores.cpu().numpy()
        doc_list = self.test_dict['381']
        with open('first500.txt', 'w') as submit_file:
            for idx in range(len(self.test_dict)):
                submit_file.write(f"{381+idx},")

                doc_score = doc_scores[idx*len(doc_list):(idx+1)*len(doc_list)]
                ranked_doc_idx = np.argsort(-doc_score)
                for didx in ranked_doc_idx[:100]:
                    doc_name = doc_list[didx]
                    submit_file.write(" " + doc_name)
                submit_file.write("\n")

        result = pd.DataFrame(columns=['Query', 'RetrievedDocuments'])
        for idx in range(len(self.test_dict)):
            doc_score = doc_scores[idx*len(doc_list):(idx+1)*len(doc_list)]
            ranked_doc_idx = np.argsort(-doc_score)

            temp = ''
            for didx in ranked_doc_idx[:100]:
                doc_name = doc_list[didx]
                temp = doc_name + ' '
            temp = pd.DataFrame(data={'Query':381+idx, 'RetrievedDocuments':[temp]}, columns=['Query', 'RetrievedDocuments'])
            result = result.append(temp,ignore_index=True)
        result.to_csv('just_bertQQ.csv', index=False)		
    
    def iteration(self, dataloader, epochs=1000, early_stop=False, patience=None, split=None):
        """
        :param dataloader: data loader for iteration
        :param epoch: total epochs for training
        :param early_stop: early_stop or not
        :param patience: patience for early_stop
        :param split: method for iteration (must be "train", "test")
        """
        
        assert split in ["train", "test"]
        
        since = time.time()
        
        if split == "train":
            batch_steps = 0
            steps = 0
            early_stop_flag=False

            self.optimizer.zero_grad()
            for epoch in range(epochs):
                for batch in dataloader:
                    batch = [*(tensor.cuda() for tensor in batch)]
                    ids, token_type, attention_mask, labels = batch
                    # Training
                    self.bert_finetune.train()
                    loss, outputs= self.bert_finetune(input_ids=ids, token_type_ids=token_type, attention_mask=attention_mask, labels=labels)
                    loss = self.criterion(outputs.squeeze(-1), labels)
                    loss.backward()
                    
                    batch_steps += 1
                    time_elapsed = time.time() - since   
                    print(f'steps {steps} | loss: {loss:.5f} | time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\r', end='')

                    if batch_steps % self.accm_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        steps += 1
                        if early_stop and steps % 200 == 0:
                            print(f'steps {steps} | Validating...                                       \r', end='')
                            val_loss, val_acc = self._val(self.val_data)
                            print(f'epochs {epoch+1} steps {steps} | val_loss: {val_loss:.5f}, val_acc: {val_acc*100:.2f}% | ' \
                                    f'time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
                            
                            self._early_stop(val_loss, val_acc)

                            if self.patience >=5:
                                early_stop_flag = True
                    
                    if early_stop_flag:
                        break
                if early_stop_flag:
                    break


                        
        if split == "test":
            self.bert_finetune.eval()
            with torch.no_grad():
                sigmoid = nn.Sigmoid()
                y_logits = np.zeros(0)

                for idx, batch in enumerate(dataloader):
                    time_elapsed = time.time() - since
                    print(f'process: {idx}/{len(dataloader)} {idx/len(dataloader)*100:0.1f}% | ' \
                            f'time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\r', end="")
                    batch = [*(tensor.cuda() for tensor in batch)]
                    ids, token_type, attention_mask = batch

                    outputs = self.bert_finetune(input_ids=ids, token_type_ids=token_type, attention_mask=attention_mask)
                    logits = outputs[0]
                    y_logits = np.concatenate([y_logits, logits.cpu().numpy().reshape(logits.shape[0])])

                y_scores = sigmoid(torch.Tensor(y_logits))
                self.save(y_scores)
            print()

                
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

def get_dataloader(train_dir, doc_dir, valid_ratio):
    # Get training data
    pos_data = pd.read_csv(train_dir + '/Pos.txt', sep=" ", names=['querys', 'documents', 'labels'])
    neg_data = pd.read_csv(train_dir + '/Neg.txt', sep=" ", names=['querys', 'documents', 'labels'])

    # split train/valid data
    raw_train_data = np.concatenate([pos_data, neg_data])
    np.random.shuffle(raw_train_data)
    val_data = raw_train_data[:int(len(raw_train_data)*valid_ratio)]
    train_data = raw_train_data[int(len(raw_train_data)*valid_ratio):]

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

    return train_dataloader, val_dataloader, test_dataloader, raw_test_dict


                                
if __name__ == '__main__':

    # Setting Variable
    train_dir_path = './data/train'
    test_dir_path = './data/test'
    doc_dir_path = './data/doc'
    valid_ratio = 0.1

    batch_size = 1
    accm_batch_size = 32

    # Setting dataloader
    train_dataloader, val_dataloader, test_dataloader, test_dict = get_dataloader(train_dir_path, test_dir_path, doc_dir_path, valid_ratio, batch_size)
    """
    # Setting BERT finetune model
    #bert_finetune = BERTFinetuneModel() # BERT Finetune的model
    bert_finetune = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)
    # Setting pretrained weights
    weight_path = './weights/0.080279.ckpt'
    
    # Create BERTTrainer
    trainer = BERTTrainer(bert_finetune=bert_finetune,
                          train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                          batch_size = batch_size, accm_batch_size = accm_batch_size, lr = 2e-5,
                          with_cuda=True)

    #trainer.train(early_stop=True, patience=5)
    torch.cuda.empty_cache()

    trainer.set_test_dataloader(test_dataloader, test_dict)
    trainer.test(load_weights=weight_path)
    
    """