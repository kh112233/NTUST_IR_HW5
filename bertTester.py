import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader

class BERTTester:
    
    def __init__(self, bert_finetune, path,
                 test_dataloader:DataLoader=None,
                 with_cuda: bool = True):
        """
        :param bert_finetune:       BERT finetune model
        :param test_dataloader:     Test dataset data loader
        :param path:                Path of result
        :param with_cuda:           training with cuda
        """
        
        # Setup cuda device for BERT model training
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        # Get BERT finetune model
        self.bert_finetune = bert_finetune
        self.bert_finetune.to(self.device)

        # Setting test dataloader
        self.test_dataloader = test_dataloader

        # Setting result path
        self.f = open(path, "a")
        
    def test(self):
        score_df = pd.DataFrame(columns = ["documents", "scores"])
        qname = ""
        sigmoid = nn.Sigmoid()

        for process, batch in enumerate(self.test_dataloader):
            ids, token_type, attention_mask, qnames, dnames = batch
            qname = qnames[0]
            ids = ids.cuda()
            token_type = token_type.cuda()
            attention_mask = attention_mask.cuda()

            # Testing
            self.bert_finetune.eval()
            with torch.no_grad():
                logits = self.bert_finetune(input_ids=ids, token_type_ids=token_type, attention_mask=attention_mask)
                scores = sigmoid(logits[0])

            scores = scores.detach().cpu().numpy().squeeze(-1)

            for dname, score in zip(dnames, scores):
                data = pd.Series({'documents':dname, 'scores':score})
                score_df = score_df.append(data, ignore_index=True)
            
            print(f"{process}/{len(self.test_dataloader)}        \r",end="")
        
        score_df = score_df.sort_values(by=['scores'], ascending=False)
        for idx in range(500):
            self.f.write(qname + " " + score_df['documents'][idx]+'\n')
        self.f.close()