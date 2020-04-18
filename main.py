import argparse

import torch

import random
import numpy as np

from data import QueryDocumentDataset, train_valid_split, train_collate_fn

# Set Seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def main(args):
    
    train_dir = args.train_dir
    test_dir = args.test_dir
    doc_dir = args.doc_dir

    ratio = args.train_valid_ratio
    batch_size = args.batch_size
     
    train = args.train

    verbose = args.verbose

    if(train == 0):
        if(verbose==2):
            print(f"Splitting the training data with {ratio*100} ratio.")
        train, valid = train_valid_split(train_dir, ratio)

        if(verbose==2):
            print("Creating dataset and dataloader.")
        train_data = QueryDocumentDataset(train_dir, doc_dir, train)
        valid_data = QueryDocumentDataset(train_dir, doc_dir, valid)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=train_collate_fn)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size*8, shuffle=False, num_workers=4, collate_fn=train_collate_fn)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Setting parser of dataset
    parser.add_argument("--train_dir", type=str, default="./data/train/" ,required=False, help="The directory of train data.")
    parser.add_argument("--test_dir", type=str, default="./data/test/" ,required=False, help="The directory of test data.")
    parser.add_argument("--doc_dir", type=str, default="./data/doc/", required=False, help="The directory of documents.")

    # Setting training parameter
    parser.add_argument("--train_valid_ratio", type=int, default=0.1, required=False, help='Ratio of train data and valid data.')
    parser.add_argument("--batch_size", type=int, default=8, required=False, help='Batch size of model.')
    parser.add_argument("--epochs", type=int, default=15, required=False, help='epochs number for train one discriminator and generator.')

    # Setting train or test flag
    parser.add_argument("--train", type=int, default=False, required=False, help='0: train, 1: test')

    # Setting test weight
    parser.add_argument("--pretrain_weight", type=str, default="", required=False, help='pretrain weight for test')

    # Setting Log type while model training or testing
    parser.add_argument("--verbose", type=int, default=2, required=False, help='log for 0: nothing, 1: valid only, 2: everything')

    main(parser.parse_args())

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
        if(verbose==2):
            print("Creating Bert Classification model.")
        bert_finetune = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)

        trainer = BERTTrainer(bert_finetune=bert_finetune,
                          train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                          batch_size = batch_size, accm_batch_size = accm_batch_size, lr = 2e-5,
                          with_cuda=True)
    
    """