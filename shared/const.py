from transformers import BertModel, BertTokenizer, OpenAIGPTModel, OpenAIGPTTokenizer
from transformers import GPT2Model, GPT2Tokenizer, CTRLModel, CTRLTokenizer
from transformers import TransfoXLModel, TransfoXLTokenizer, XLNetModel, XLNetTokenizer, DistilBertModel, DistilBertTokenizer
from transformers import RobertaModel, RobertaTokenizer, XLMRobertaModel, XLMRobertaTokenizer


task_ner_labels = {
    'conll03': ['ORG', 'MISC', 'PER', 'LOC'],
    'twitter': ['geoloc', 'facility', 'movie', 'company', 'product', 'person', 'other', 'sportsteam', 'tvshow', 'musicartist'],
    'bc5cdr': ['Disease', 'Chemical'], 
    'wiki': ['ORG', 'MISC', 'PER', 'LOC'],
}

PLMs = {
    'dmis-lab/biobert-base-cased-v1.1' : {  "model": BertModel,  "tokenizer" : BertTokenizer},
    'bert-base-uncased' : {  "model": BertModel,  "tokenizer" : BertTokenizer},
    'bert-base-cased' : {  "model": BertModel,  "tokenizer" : BertTokenizer},
    'bert-large-cased' : {  "model": BertModel,  "tokenizer" : BertTokenizer},
    'openai-gpt': {"model": OpenAIGPTModel, "tokenizer": OpenAIGPTTokenizer},
    'gpt2': {"model": GPT2Model, "tokenizer": GPT2Tokenizer},
    'ctrl': {"model": CTRLModel, "tokenizer": CTRLTokenizer},
    'transfo-xl-wt103': {"model": TransfoXLModel, "tokenizer": TransfoXLTokenizer},
    'xlnet-base-cased': {"model": XLNetModel, "tokenizer": XLNetTokenizer},
    'distilbert-base-cased': {"model": DistilBertModel, "tokenizer": DistilBertTokenizer},
    'roberta-base': {"model": RobertaModel, "tokenizer": RobertaTokenizer},
    'roberta-large': {"model": RobertaModel, "tokenizer": RobertaTokenizer},
    'xlm-roberta-base': {"model": XLMRobertaModel, "tokenizer": XLMRobertaTokenizer},
}

def get_labelmap(label_list):
    label2id = {}
    id2label = {}
    for i, label in enumerate(label_list):
        label2id[label] = i + 1
        id2label[i + 1] = label
    return label2id, id2label
