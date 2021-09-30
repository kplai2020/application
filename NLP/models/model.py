
"""

Main class of the model architecture.

"""

import torch.nn as nn
from pytorch_pretrained_bert import BertModel

class BioBERTModel(nn.Module):

    """
    Setup a model for healthcare-related NER detection.
    """

    def __init__(self, n_class, config_path, state_dict):
        """
        Args:
            n_class: (int) number of out feaatures.
            config_path: (str) pretrained model file path.
            state_dict: (dict) model states params.
        """
        super().__init__()
        self.bert = BertModel.from_pretrained(config_path)
        self.bert.load_state_dict(state_dict, strict=False)
        self.dropout = nn.Dropout(p=0.3)
        self.output = nn.Linear(self.bert.config.hidden_size, n_class)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, input_ids, attention_mask):
        encoded_layer, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        encl = encoded_layer[-1]
        out = self.dropout(encl)
        out = self.output(out)
        return out, out.argmax(-1)


