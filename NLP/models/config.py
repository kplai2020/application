"""

All settings and configurations to run the scripts can be applied in this file. 
Settings related to input data, file exports and model configuration will be available in this script.

"""


# data files
TRAIN_FILE = "./data/train.tsv"
TEST_FILE = "./data/test.tsv" # use for evaluation

# model files
TRAINED_MODEL_FILE = './results/trained_model.pt'
NER_RESULTS_FILE = './results/ner_result.csv'

# torch setting
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BioBERT base
import transformers
BERT_CONFIG_FILE = './biobert_v1.1_pubmed' # /config.json'
TOKENIZER = transformers.BertTokenizer(vocab_file='biobert_v1.1_pubmed/vocab.txt', do_lower_case=False)
WEIGHTS_BIN = torch.load('./biobert_v1.1_pubmed/pytorch_model.bin', map_location=DEVICE)

# params config
MAX_LEN = 75
MAX_GRAD_NORM = 1.0
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 8
EPOCHS = 1

