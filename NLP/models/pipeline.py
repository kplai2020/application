"""

A pipeline to use the trained model to detect entities(NER).

Things to do:
1. applying the model:
    - Input: str a sentence
    - Output: 
      - List: a list of entities if you are doing NER

"""

import pandas as pd
from collections import OrderedDict

import config
from model import BioBERTModel
from data_prep import NERDataset
import train

import nltk
import torch
import torch.nn as nn

import transformers

# set params
tokenizer = config.TOKENIZER
ner_results_file = config.NER_RESULTS_FILE

# model setup
test_file = config.TEST_FILE
ner_pred_data = NERDataset(test_file)
t_vals = ner_pred_data.tags_vals
n_tags = len(ner_pred_data.tag2idx)
bert_config_file = config.BERT_CONFIG_FILE
state_dict = train.weights_getter()

model = BioBERTModel(n_tags, bert_config_file, state_dict)

# load pre_trained 
TRAINED_MODEL_FILE = config.TRAINED_MODEL_FILE
model.load_state_dict(torch.load(TRAINED_MODEL_FILE))

def tokenize_and_preserve(sentence):

    tokenized_sentence = []
    for word in sentence:
        tokenized_word = tokenizer.tokenize(word)   
        tokenized_sentence.extend(tokenized_word)

    return tokenized_sentence

def act_pred_comparison_list(input_ids, input_attentions):

    """
    Clean data and predict results. 
    """
    
    actual_sentences = []
    pred_labels = []

    for x,y in zip(input_ids,input_attentions):
        x = torch.tensor(x)
        y = torch.tensor(y)
        x = x.view(-1,x.size()[-1])
        y = y.view(-1,y.size()[-1])
        with torch.no_grad():
            _,y_hat = model(x,y)
        label_indices = y_hat.to('cpu').numpy()
        
        tokens = tokenizer.convert_ids_to_tokens(x.to('cpu').numpy()[0])
        new_tokens, new_labels = [], []
        for token, label_idx in zip(tokens, label_indices[0]):
            if token.startswith("##"):
                new_tokens[-1] = new_tokens[-1] + token[2:]
            else:
                new_labels.append(t_vals[label_idx])
                new_tokens.append(token)
        actual_sentences.append(new_tokens)
        pred_labels.append(new_labels)
    
    return actual_sentences, pred_labels


def main_run(text):

    """
    Perform NER detection via user inputs and store result in .csv.

    Args:
        text: (str or list) inputs by the user.
    
    NOTE: For testing purpose, this script provides a few different types of texts for a quick check.
    Texts can be found on the bottom of the script.

    """

    sent_text = nltk.sent_tokenize(text)
    tokenized_text = []
    for sentence in sent_text:
        tokenized_text.append(nltk.word_tokenize(sentence))

    tok_texts = [tokenize_and_preserve(sent) for sent in tokenized_text]

    input_ids = [tokenizer.convert_tokens_to_ids(txt) for txt in tok_texts]
    input_attentions = [[1]*len(in_id) for in_id in input_ids]

    actual_sentences, pred_labels = act_pred_comparison_list(input_ids, input_attentions)

    ner_collections = []
    cnt_det = 0 
    for token, label in zip(actual_sentences, pred_labels):
        for t,l in zip(token,label):
            ner_collections.append([t,l])
            if l == 'B' or l == 'I':
                cnt_det += 1
                ner_collections.append([t,l])
                print('\n\nDetected:\n\n')
                print("{}\t{}".format(t, l))

    print(ner_collections) # checking purpose
    ner_result = pd.DataFrame(ner_collections, columns=['Word', 'Tag'])
    ner_result.to_csv(ner_results_file)
    print(f'\n\n{cnt_det} NER is(are) detected.\n\n')
    print('\n\nAll texts and detected labels can be found in ner_result.csv.\n\n')


if __name__ == '__main__':

    userInput = input("Enter a sentence(s): ")
    print("\n\nDetecting...\n\n")
    main_run(userInput)
    


# ======================= Checking Samples ======================= #

'''
text = """HisG. Escherichia. HisG. panadol actifast"""

text = """
In addition to their essential catalytic role in protein biosynthesis, aminoacyl-tRNA synthetases participate in numerous other functions, including regulation of gene expression and amino acid biosynthesis via transamidation pathways. 
Herein, we describe a class of aminoacyl-tRNA synthetase-like (HisZ) proteins based on the catalytic core of the contemporary class II histidyl-tRNA synthetase whose members lack aminoacylation activity but are instead essential components of the first enzyme in histidine biosynthesis ATP phosphoribosyltransferase (HisG). 
Prediction of the function of HisZ in Lactococcus lactis was assisted by comparative genomics, a technique that revealed a link between the presence or the absence of HisZ and a systematic variation in the length of the HisG polypeptide. 
HisZ is required for histidine prototrophy, and three other lines of evidence support the direct involvement of HisZ in the transferase function. 
(i) Genetic experiments demonstrate that complementation of an in-frame deletion of HisG from Escherichia coli (which does not possess HisZ) requires both HisG and HisZ from L. lactis. 
(ii) Coelution of HisG and HisZ during affinity chromatography provides evidence of direct physical interaction. 
(iii) Both HisG and HisZ are required for catalysis of the ATP phosphoribosyltransferase reaction.
"""
text = """
    test,[]ing
    """
    
text = ["Individual atoms can be held together by chemical bonds to form molecules and ionic compounds.", "A hydrogen bond is primarily an electrostatic force of attraction between a hydrogen atom which is covalently bound to a more electronegative atom or group such as oxygen."]

'''

# ======================= ================ ======================= #
