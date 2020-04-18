import time
import datetime
import os
import copy

import numpy as np
import pandas as pd
import random

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader


class BERTTrainer:
    
    def __init__(self, bert_finetune,
                 train_dataloader:DataLoader=None, valid_dataloader=None,
                 batch_size = 2, accm_batch_size = 32, lr = 2e-5,
                 verbose = 2, with_cuda: bool = True):
        """
        :param bert_finetune:       BERT finetune model
        :param train_dataloader:    train dataset data loader
        :param val_dataloader:      valid dataset data loader
        :param batch_size:          batch size of data loader 
        :param accm_batch_size:     accumulate gradient batch size
        :param lr:                  learning rate of AdamW
        :param verbose:             Log type
        :param with_cuda:           training with cuda
        """
        
        # Setup cuda device for BERT model training
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        # Setup Log type
        self.verbose = verbose

        # Get BERT finetune model
        self.bert_finetune = bert_finetune
        self.bert_finetune.to(self.device)

        # Setting the AdamW optimizer with hyper-parameter
        self.optimizer = optim.AdamW(self.bert_finetune.parameters(), lr=lr) # suggest lr: 2e-5~3e-5

        # Setting train dataloader
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        # Using binary cross entropy with logists loss for criterion
        self.criterion = nn.BCEWithLogitsLoss()

        # Setting accmulate gradient
        self.accm_batch_size = accm_batch_size
        self.batch_size = batch_size
        assert self.accm_batch_size % self.batch_size == 0
        self.accm_steps = self.accm_batch_size // self.batch_size

        # Setting patience of early stop
        self.patience = 0

        # Setting the path that saving weights
        self.save_path = "./weights/"
        
    def _valid_function(self):
        self.bert_finetune.eval() 
        with torch.no_grad():
            sigmoid = nn.Sigmoid()
            y_logits, y_true = np.zeros(0), np.zeros(0)

            for batch in self.valid_dataloader:
                
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
    
    def train(self, epochs=100, early_stop="None"):
        """
        :param epoch:       Total epochs for training.
        :param early_stop:  The type of early stop(BCE, ACC, None).
        """

        assert early_stop in ['BCE', 'ACC', 'None']
        best_loss = 100
        best_metrics = 0

        since = time.time()
        batch_steps = 0
        steps = 0
        patience = 0
        early_stop_flag=False

        self.optimizer.zero_grad()
        for epoch in range(epochs):
            for batch in self.train_dataloader:
                batch = [*(tensor.cuda() for tensor in batch)]
                ids, token_type, attention_mask, labels = batch

                # Training
                self.bert_finetune.train()
                loss, outputs= self.bert_finetune(input_ids=ids, token_type_ids=token_type, attention_mask=attention_mask, labels=labels)
                outputs = outputs.squeeze(-1)
                loss = self.criterion(outputs, labels)
                loss.backward()
                
                batch_steps += 1
                time_elapsed = time.time() - since   

                #if(self.verbose == 2):
                    #print(f'batch steps {batch_steps} | loss: {loss:.5f} | time: {datetime.timedelta(seconds=(int)(time_elapsed))}\r', end='')
                
                # accumulated batch
                if batch_steps % self.accm_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    steps += 1
                    if(self.verbose == 2):
                        print(f'steps {steps} | loss: {loss:.5f} | time: {datetime.timedelta(seconds=(int)(time_elapsed))}\r', end='')
                    
                    if(steps % 200 == 0 and early_stop!="None"):

                        if(self.verbose == 1 or self.verbose == 2):
                            print("Validating.....                                                                  \r", end='')
                        val_loss, val_metrics = self._valid_function()

                        if(self.verbose == 1 or self.verbose == 2):
                            print(f"steps {steps} | val_loss: {val_loss:.5f} | val_acc: {val_metrics:.5f} | time: {datetime.timedelta(seconds=(int)(time_elapsed))}")
                            
                        # Earlystopping
                        if(early_stop == "BCE"):
                            path = self.save_path + "BCE/"

                            if val_loss < best_loss:
                                best_loss = val_loss

                                if not os.path.exists(path):
                                    os.mkdir(path)

                                best_weight = copy.deepcopy(self.bert_finetune.state_dict()) 
                                torch.save(best_weight, path + f"MSE:{best_loss:.5f}.ckpt")
                                patience = 0
                            else:
                                patience+=1

                            if (self.verbose == 1 or self.verbose == 2):
                                print(f'steps {steps} | best_BCE: {best_loss:.5f}, patience: {patience:.0f}')

                        elif(early_stop == "ACC"):
                            path = self.save_path + "ACC/"

                            if val_metrics > best_metrics:
                                best_metrics = val_metrics
                                
                                if not os.path.exists(path):
                                    os.mkdir(path)
                                    
                                best_weight = copy.deepcopy(self.bert_finetune.state_dict()) 
                                torch.save(best_weight, path + f"ACC:{best_metrics:.5f}.ckpt")
                                patience = 0
                            else:
                                patience+=1

                            if (self.verbose == 1 or self.verbose == 2):
                                print(f'steps {steps} | best_ACC: {best_metrics:.5f}, patience: {patience:.0f}')

                    if patience >= 5:
                        early_stop_flag = True
                
                if early_stop_flag:
                    break
            if early_stop_flag:
                break