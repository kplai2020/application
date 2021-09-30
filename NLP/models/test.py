
"""

Model evaluation on test set.

"""


import numpy as np
import torch

import config
import train
from data_prep import NERDataset
from model import BioBERTModel

import numpy as np

import torch
import torch.nn as nn


def model_eval(model,data_loader,loss_fn,device):
    model = model.eval()
    
    losses = []
    correct_predictions = 0
    
    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
        
            outputs,y_hat = model(b_input_ids,b_input_mask)
        
            _,preds = torch.max(outputs,dim=2)
            outputs = outputs.view(-1,outputs.shape[-1])
            b_labels_shaped = b_labels.view(-1)
            loss = loss_fn(outputs,b_labels_shaped)
            correct_predictions += torch.sum(preds == b_labels)
            losses.append(loss.item())
    
    return correct_predictions.double()/len(data_loader) , np.mean(losses)


def test_main():

    # set params
    test_file = config.TEST_FILE
    test_batch_size = config.TRAIN_BATCH_SIZE # treat train and test batch size equally
    max_len = config.MAX_LEN
    epochs = config.EPOCHS
    device = config.DEVICE

    # load data
    ner_test_data = NERDataset(test_file)
    test_dataloader = ner_test_data.load_test_data()

    # model setup
    n_tags = len(ner_test_data.tag2idx)
    bert_config_file = config.BERT_CONFIG_FILE
    state_dict = train.weights_getter()
    model = BioBERTModel(n_tags, bert_config_file, state_dict)
    loss_fn = nn.CrossEntropyLoss().to(device)

    # load pre_trained 
    TRAINED_MODEL_FILE = config.TRAINED_MODEL_FILE
    model.load_state_dict(torch.load(TRAINED_MODEL_FILE))

    # track performance params
    normalizer = test_batch_size * max_len

    for epoch in range(epochs):

        print(f'======== Epoch {epoch+1}/{epochs} ========')
        te_acc,te_loss = model_eval(model,test_dataloader,loss_fn,device)
        te_acc = te_acc/normalizer
        print(f'Val Loss: {te_loss} Val Accuracy: {te_acc}')



if __name__ == '__main__':

    test_main()
    print('\n\nRunning completed\n\n')
    

        
    