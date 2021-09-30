"""

All data related preparations will be grouped into this script.

"""

import config
import pandas as pd

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

class NERDataset(object):
    
    """
    Handles all aspect of the data. 
    """

    max_len = config.MAX_LEN
    batch_size = config.TRAIN_BATCH_SIZE

    # biobert base
    TOKENIZER = config.TOKENIZER

    def __init__(self, data_path):

        self.df = pd.read_csv(data_path, sep='\t', names=['Word', 'Tag'], header=None).fillna(method ="ffill")
        self.df = self.df[0:5000]
        self.df['Sentence #'] = (self.df['Word']=='.').shift(0).cumsum()
        agg_func = lambda s: [(w, t) for w, t in zip(s["Word"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.df.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]
        self.tags_vals = list(set(self.df["Tag"].values))
        self.tags_vals.append("PAD")

        self.tag2idx = {t: i for i, t in enumerate(self.tags_vals)}

    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

    def tok_with_labels(self, sent, text_labels):
        """tokenize with labels intact"""
        tok_sent = []
        labels = []
        for word, label in zip(sent, text_labels):
            tok_word = self.TOKENIZER.tokenize(word)
            n_subwords = len(tok_word)
            tok_sent.extend(tok_word)
            labels.extend([label] * n_subwords)
        return tok_sent, labels

    def tokenize_label_pad_sequences(self, max_len):
        """
        Pad sequences to the same length.
        
        Args:
            max_len: (int) define sentence length.
        
        Returns:
            inputs_ids: (array) list of words IDs 
            tags: (array) list of words Tags
        """

        words = [[word[0] for word in sentence] for sentence in self.sentences]
        labels = [[tag[1] for tag in sentence] for sentence in self.sentences]

        tok_texts_and_labels = [
            self.tok_with_labels(word, labs)
            for word, labs in zip(words, labels)
        ]

        tok_texts = [tok_label_pair[0] for tok_label_pair in tok_texts_and_labels]
        labels = [tok_label_pair[1] for tok_label_pair in tok_texts_and_labels]
        
        self.input_ids = pad_sequences([self.TOKENIZER.convert_tokens_to_ids(txt) for txt in tok_texts],
                                        maxlen=max_len, dtype="long", value=0.0,
                                        truncating="post", padding="post")
                                        
        self.tags = pad_sequences([[self.tag2idx.get(l) for l in lab] for lab in labels],
                                    maxlen=max_len, value=self.tag2idx["PAD"], padding="post",
                                    dtype="long", truncating="post")

        return self.input_ids, self.tags


    def load_training_data(self):
        """
        Loads and cleans data for training and validation.
        """
        self.tokenize_label_pad_sequences(self.max_len)

        attention_masks = [[float(i != 0.0) for i in ii] for ii in self.input_ids]
        tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(self.input_ids, self.tags,
                                                            random_state=22, test_size=0.1)
        tr_masks, val_masks, _, _ = train_test_split(attention_masks, self.input_ids,
                                                    random_state=22, test_size=0.1)
        
        tr_inputs = torch.tensor(tr_inputs)
        tr_masks = torch.tensor(tr_masks)
        tr_tags = torch.tensor(tr_tags)
        train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)

        val_inputs = torch.tensor(val_inputs)
        val_masks = torch.tensor(val_masks)
        val_tags = torch.tensor(val_tags)
        valid_data = TensorDataset(val_inputs, val_masks, val_tags)
        valid_sampler = SequentialSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=self.batch_size)

        return train_dataloader, valid_dataloader
    
    def load_test_data(self):
        """
        Loads and cleans data for evaluation.
        """

        self.tokenize_label_pad_sequences(self.max_len)

        te_masks = [[float(i != 0.0) for i in ii] for ii in self.input_ids]

        te_inputs = torch.tensor(self.input_ids)
        te_masks = torch.tensor(te_masks)
        te_tags = torch.tensor(self.tags)
        te_data = TensorDataset(te_inputs, te_masks, te_tags)
        te_sampler = SequentialSampler(te_data)
        te_dataloader = DataLoader(te_data, sampler=te_sampler, batch_size=self.batch_size)

        return te_dataloader
