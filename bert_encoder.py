import torch.nn as nn
import torch
import math
from transformers import BertForMaskedLM as WoBertForMaskedLM
from transformers import BertConfig
from src.wobert import WoBertTokenizer
from torch.optim import AdamW
import jieba
import string
import torch.nn.functional as F

from torch.utils.data import DataLoader
jieba.load_userdict('/home/wushuang/Sysu/bert/bert_training/vocab/my_vocab.txt')
stop_words = []
with open('stop_word.txt','r',encoding='utf-8') as f:
    for line in f.readlines():
        stop_words.append(line.strip())
def pre_text(text):
	split_text = jieba.cut(text)
	ntext= []
	for w in split_text:
		if w not in stop_words and w.strip() != '\n':
			ntext.append(w)
	return "".join(ntext)


import json
import numpy as np
from tqdm import tqdm
@torch.no_grad()
def bert_encode(bert,text):
    bert.eval()
    path = '/home/wushuang/Sysu/bert/bert_training/vocab/my_vocab.txt'
    tokenizer = WoBertTokenizer.from_pretrained(path)
    devices = torch.device('cuda:1')
    encode_text=text[:250]+text[-250:]

    features = tokenizer([pre_text(encode_text)], max_length=512, truncation=True, padding=True, return_tensors='pt')
    input_ids,mask = (features['input_ids'].to(devices),features['attention_mask'].to(devices))
    y_=bert.encode(input_ids,mask)
    return y_
