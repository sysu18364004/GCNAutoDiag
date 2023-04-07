
import numpy as np
from math import log
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import torch.nn as nn


def get_weights_hidden(model,features,adj,adj_lab,train_size,test_size,vocab_length,lab_length,idx,lambda1):
    model.eval()
    hidden_output, hidden_output_lab,label_embedding = model.encode(features, adj,adj_lab,idx,lambda1)
    weights1 = model.gc1.weight[:vocab_length]
    bias1 = model.gc1.bias
    weights1.require_grad = False
    bias1.require_grad = False
    weights2 = model.gc2.weight[:vocab_length]
    bias2 = model.gc2.bias
    weights2.require_grad = False
    bias2.require_grad = False

    weights1_lab = model.gc_lab1.weight[:lab_length]
    bias1_lab = model.gc_lab1.bias
    weights1_lab.require_grad = False
    bias1_lab.require_grad = False
    weights2_lab = model.gc_lab2.weight[:lab_length]
    bias2_lab = model.gc_lab2.bias
    weights2_lab.require_grad = False
    bias2_lab.require_grad = False

    fc = model.fc
    fc.require_grad = False
    return [hidden_output[train_size+test_size:train_size+test_size+vocab_length],hidden_output_lab[train_size+test_size:train_size+test_size+lab_length],label_embedding, weights1, bias1, weights2, bias2,weights1_lab,bias1_lab,weights2_lab,bias2_lab,fc]

def normalize_edge(tokenized_test_edge,norm_item,train_size,test_size,vocab_length):
    word_doc, one = tokenized_test_edge[:,:-1],tokenized_test_edge[:,-1]
    rowsum = tokenized_test_edge.sum(1)
    word_doc_sqrt = np.power(rowsum, -0.5).flatten()
    word_doc_sqrt[np.isinf(word_doc_sqrt)] = 0.
    word_doc_sqrt = word_doc_sqrt.reshape(1,-1)
    word_diag = norm_item[train_size+test_size:train_size+test_size+vocab_length].reshape(1,-1)
    normal_p = word_doc_sqrt.T.dot(word_diag)
    test_edges = sp.coo_matrix(word_doc)
    test_edges_emb = test_edges.multiply(normal_p)
    another_cal_ = test_edges_emb.toarray()
    tokenized_test_edge1 = np.concatenate([another_cal_,(word_doc_sqrt**2).reshape(-1,1)],axis=1)
    return tokenized_test_edge1
from bert_encoder import bert_encode
def get_test_emb(tokenize_test_sentences, word_id_map, vocab_length, word_doc_freq, word_list,lab_id_map,lab_length,lab_doc_freq,lab_list, train_size,norm_item,norm_item_lab,bert):
#     norm_item = norm_item[:vocab_length]
    test_size = len(tokenize_test_sentences)
    test_emb = [[0]*vocab_length for _ in range(test_size)]
    test_emb = []
    

    tokenized_test_edge = [[0]*vocab_length+[1] for _ in range(test_size)]
    tokenized_test_edge_lab = [[0]*lab_length+[1] for _ in range(test_size)]
    for i in range(test_size):
        tokenized_test_sample = tokenize_test_sentences[i]
        test_emb.append(bert_encode(bert,''.join(tokenized_test_sample[0]))[0].cpu().numpy())
        word_freq_list = [0]*vocab_length
        for word in tokenized_test_sample[0]:
            if word in word_id_map:
                word_freq_list[word_id_map[word]]+=1
            
        for word in tokenized_test_sample[0]:
            if word in word_id_map:
                j = word_id_map[word]
                freq = word_freq_list[j]   

                idf = log(1.0 * train_size / (word_doc_freq[word_list[j]]+1))
                w = idf
                tokenized_test_edge[i][j] = w
        lab_freq_list = [0]*lab_length
        for word in tokenized_test_sample[1]:
            if word in lab_id_map:
                lab_freq_list[lab_id_map[word]]+=1
            
        for word in tokenized_test_sample[1]:
            if word in lab_id_map:
                j = lab_id_map[word]
                idf = log(1.0 * train_size / (lab_doc_freq[lab_list[j]]+1))
                w = idf
                tokenized_test_edge_lab[i][j] = w

    tokenized_test_edge = np.array(tokenized_test_edge)
    tokenized_test_edge = normalize_edge(tokenized_test_edge,norm_item,train_size,test_size,vocab_length)
    tokenized_test_edge_lab = np.array(tokenized_test_edge_lab)
    tokenized_test_edge_lab = normalize_edge(tokenized_test_edge_lab,norm_item_lab,train_size,test_size,lab_length)
    return test_emb, tokenized_test_edge,tokenized_test_edge_lab

def normal_vec(t):
    norm = (t - t.mean(1).unsqueeze(1))/t.std(1).unsqueeze(1)
    norm[torch.isnan(norm)] = 0.0
    return norm


@torch.no_grad()
def test_model(model, test_emb, tokenized_test_edge,tokenized_test_edge_lab,model_weights_list,device,lambda1):

    hidden_output,hidden_output_lab,label_embedding, weights1, bias1, weights2, bias2,weights1_lab, bias1_lab, weights2_lab, bias2_lab,fc = model_weights_list
    test_result = []
    test_size = len(tokenized_test_edge[0])
    for ind in range(len(test_emb)):
        tokenized_test_edge_temp = torch.FloatTensor([tokenized_test_edge[ind]]).to(device)
        tokenized_test_edge_temp_lab = torch.FloatTensor([tokenized_test_edge_lab[ind]]).to(device)

        hidden_temp = torch.FloatTensor([test_emb[ind].tolist()]).to(device)
        hidden_temp_ = F.relu(torch.mm(tokenized_test_edge_temp, torch.cat((weights1, hidden_temp))) + bias1)
        hidden_temp_lab = F.relu(torch.mm(tokenized_test_edge_temp_lab, torch.cat((weights1_lab, hidden_temp))) + bias1_lab)
        hidden_temp = hidden_temp_ + hidden_temp_lab


        test_hidden_temp = torch.cat((hidden_output,hidden_temp))
        test_output_temp = torch.mm(tokenized_test_edge_temp, torch.mm(test_hidden_temp, weights2)) + bias2

        test_hidden_temp_lab = torch.cat((hidden_output_lab,hidden_temp_lab))
        test_output_temp_lab = torch.mm(tokenized_test_edge_temp_lab, torch.mm(test_hidden_temp_lab, weights2_lab)) + bias2_lab
        test_output_temp = test_output_temp+ test_output_temp_lab*lambda1
        test_output_temp = F.relu(test_output_temp)
        predict_temp = (fc(test_output_temp)).cpu() 
        test_result.append(torch.sigmoid(predict_temp).cpu())

    return test_result

