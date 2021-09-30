"""

Model training on train data (further splitting into Validation test). 

"""

import config
import test
from model import BioBERTModel
from data_prep import NERDataset 

import numpy as np
from collections import defaultdict, OrderedDict

import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup


def weights_getter():
    """get pretrained model."""
    try:
        tmp_d = config.WEIGHTS_BIN # './biobert_v1.1_pubmed/pytorch_model.bin'
        state_dict = OrderedDict()
        for i in list(tmp_d.keys())[:199]:
            x = i
            if i.find('bert') > -1:
                x = '.'.join(i.split('.')[1:])
            state_dict[x] = tmp_d[i]
        return state_dict

    except ValueError:
        print('Do ensure you have the weights file ready.')


def optimizer_setter(model, FINE_TUNING=True):

    if FINE_TUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_params = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_params = [{"params": [p for n, p in param_optimizer]}]

    optim = AdamW(optimizer_params, lr=3e-5, eps=1e-8)

    return optim

# assign a para only for train_epoch() 
max_grad_norm = config.MAX_GRAD_NORM
def train_epoch(model,data_loader,loss_fn,optimizer,device,scheduler):

    model = model.train()
    losses = []
    correct_predictions = 0
    for step,batch in enumerate(data_loader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        outputs,y_hat = model(b_input_ids,b_input_mask)
        
        _,preds = torch.max(outputs,dim=2)
        outputs = outputs.view(-1,outputs.shape[-1])
        b_labels_shaped = b_labels.view(-1)
        loss = loss_fn(outputs,b_labels_shaped)
        correct_predictions += torch.sum(preds == b_labels)
        losses.append(loss.item())
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
    return correct_predictions.double()/len(data_loader) , np.mean(losses)


def train_main():
    
    # set params    
    device = config.DEVICE
    bert_config_file = config.BERT_CONFIG_FILE #'./biobert_v1.1_pubmed/config.json' #
    train_file = config.TRAIN_FILE

    train_batch_size = config.TRAIN_BATCH_SIZE
    max_len = config.MAX_LEN
    epochs = config.EPOCHS

    TRAINED_MODEL_FILE = config.TRAINED_MODEL_FILE

    # load data
    ner_train_data = NERDataset(train_file)
    train_dataloader, valid_dataloader = ner_train_data.load_training_data()   

    # model setup
    n_tags = len(ner_train_data.tag2idx)
    state_dict = weights_getter()
    model = BioBERTModel(n_tags, bert_config_file, state_dict)
    optimizer = optimizer_setter(model, FINE_TUNING=True)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    loss_fn = nn.CrossEntropyLoss().to(device)

    # track performance params
    history = defaultdict(list)
    normalizer = train_batch_size * max_len
    loss_values = []

    for epoch in range(epochs):
        
        total_loss = 0
        print(f'======== Epoch {epoch+1}/{epochs} ========')
        train_acc,train_loss = train_epoch(model,train_dataloader,loss_fn,optimizer,device,scheduler)
        train_acc = train_acc/normalizer
        print(f'Train Loss: {train_loss} Train Accuracy: {train_acc}')
        total_loss += train_loss.item()
        
        avg_train_loss = total_loss / len(train_dataloader)  
        loss_values.append(avg_train_loss)
        
        val_acc,val_loss = test.model_eval(model,valid_dataloader,loss_fn,device)
        val_acc = val_acc/normalizer
        print(f'Val Loss: {val_loss} Val Accuracy: {val_acc}')
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
    
    # save trained model params
    torch.save(model.state_dict(), TRAINED_MODEL_FILE)


if __name__ == '__main__':

    train_main()
    print('\n\nRunning completed\n\n')



    
