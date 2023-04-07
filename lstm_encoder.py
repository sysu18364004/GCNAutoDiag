import torch
import re

def clean_str(string):
    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_tsv(filename,rate=1):
    labels = []
    sentences = []
    

    with open(filename,'r',encoding='utf-8') as f:
        lines = f.readlines()
        random.shuffle(lines)
        length = int(rate * len(lines))
        for line in lines[:length]:
            # print(line.split())
            label,*word = line.split()

            labels.append(list(map(int,label)))
            sentences.append(" ".join(word))

    return sentences,labels
import math
import torch.nn.functional as F
def attention_net(x, query, mask=None): 
        
        d_k = query.size(-1)     # d_k为query的维度
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)  
        alpha_n = F.softmax(scores, dim=-1) 
        context = torch.matmul(alpha_n, x).sum(1)
        
        return context, alpha_n

class Tokenizier():
    def __init__(self,dic):
        self.dic = dic
    def single_sentence(self,sentence):
        sentence = clean_str(sentence)
        input_ids = []
        for word in sentence.split():
            if word in self.dic:
                # print(word)
                input_ids.append(self.dic[word]+1)
        return input_ids
    def encode(self,sentences):
        ans = {}
        if isinstance(sentences,str):
            ans['input_ids'] = self.single_sentence(sentences)
            attention_mask = [1]*len(input_ids)
            ans['attention_mask'] = attention_mask
            return ans
        
        input_ids = []
        max_l = 0
        for sentence in sentences:
            input_ids_item = self.single_sentence(sentence)
            input_ids.append(input_ids_item)
            max_l = max_l if max_l > len(input_ids_item) else len(input_ids_item)
        
        padding_input_ids = []
        attention_mask = []
        for input_id in input_ids:
            padding_input_ids.append(input_id+[0]*(max_l - len(input_id)))
            attention_mask.append([1]*len(input_id)+[0]*(max_l - len(input_id)))
        ans['input_ids'] = padding_input_ids
        ans['attention_mask'] = attention_mask
        return ans
import json

with open('/home/wushuang/Sysu/ipynb/code/codeTextClaassification/LSTM_remove/dict_ehr.json','r',encoding='utf-8') as f:
        wordid_map = json.load(f)
tokenizier = Tokenizier(wordid_map)
@torch.no_grad()
def lstm_encode(text,info):
    embeding = info['embeding']
    lstm = info['lstm']
    inputs = tokenizier.encode([text])['input_ids']
    inputs = torch.LongTensor(inputs).to('cuda:1')
    out = lstm(embeding(inputs.permute(1,0)))[0].permute(1,0,2)
    attn_output, alpha_n = attention_net(out, out)
    return attn_output[0]


