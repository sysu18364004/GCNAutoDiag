from tqdm.auto import tqdm
import scipy.sparse as sp
from math import log
import numpy as np
import torch.nn as nn
import torch
from bert_encoder import bert_encode
def ordered_word_pair(a, b):
    if a > b:
        return b, a
    else:
        return a, b
def get_label_nums(label_item,base):
    ans = []
    for i,f in enumerate(label_item):
        if f == 1:
            ans.append(base+i)
    return ans
def get_weight(word_list,word_id_map,labels,train_size,test_size,tokenize_sentences,args,bert,order=1,need_window=True,need_init=True):
    window_size = 20
    total_W = 0
    word_occurrence = {}
    word_pair_occurrence = {}
    vocab_length = len(word_list)

    label_length = len(labels[0])
    node_size = train_size + len(word_list) + label_length + test_size
    def update_word_and_word_pair_occurrence(q):
        unique_q = list(set(q))
        for i in unique_q:
            try:
                word_occurrence[i] += 1
            except:
                word_occurrence[i] = 1
        for i in range(len(unique_q)):
            for j in range(i+1, len(unique_q)):
                word1 = unique_q[i]
                word2 = unique_q[j]
                word1, word2 = ordered_word_pair(word1, word2)
                try:
                    word_pair_occurrence[(word1, word2)] += 1
                except:
                    word_pair_occurrence[(word1, word2)] = 1
    if not args.easy_copy:
        print("Calculating PMI")
    for ind in range(train_size):
        words = tokenize_sentences[ind][order]
        word_unique = list(map(lambda x : word_id_map[x],list(set(words))))
        update_word_and_word_pair_occurrence(get_label_nums(labels[ind],vocab_length)+word_unique)
        if not need_window:
            total_W += 1
            continue
        q = []
        # push the first (window_size) words into a queue
        for i in range(min(window_size, len(words))):
            q += [word_id_map[words[i]]]
        # update the total number of the sliding windows
        total_W += 1
        # update the number of sliding windows that contain each word and word pair
        update_word_and_word_pair_occurrence(q)

        now_next_word_index = window_size
        # pop the first word out and let the next word in, keep doing this until the end of the document
        while now_next_word_index<len(words):
            q.pop(0)
            q += [word_id_map[words[now_next_word_index]]]
            now_next_word_index += 1
            # update the total number of the sliding windows
            total_W += 1
            # update the number of sliding windows that contain each word and word pair
            update_word_and_word_pair_occurrence(q)

    # calculate PMI for edges
    row = []
    col = []
    weight = []
    for word_pair in word_pair_occurrence:
        i = word_pair[0]
        j = word_pair[1]
        count = word_pair_occurrence[word_pair]
        word_freq_i = word_occurrence[i]
        word_freq_j = word_occurrence[j]
        pmi = log((count * total_W) / (word_freq_i * word_freq_j)) 
        if pmi <=0:
            continue
        row.append(train_size + test_size+i)
        col.append(train_size + test_size+j)
        weight.append(pmi)
        row.append(train_size + test_size+j)
        col.append(train_size + test_size+i)
        weight.append(pmi)
    if not args.easy_copy:
        print("PMI finished.")


    #get each word appears in which document
    word_doc_list = {}
    for word in word_list:
        word_doc_list[word]=[]

    for i in range(train_size):
        doc_words = tokenize_sentences[i][order]
        unique_words = set(doc_words)
        for word in unique_words:
            exsit_list = word_doc_list[word]
            exsit_list.append(i)
            word_doc_list[word] = exsit_list

    #document frequency
    word_doc_freq = {}
    for word, doc_list in word_doc_list.items():
        word_doc_freq[word] = len(doc_list)

    # term frequency
    doc_word_freq = {}

    for doc_id in range(train_size+test_size):
    # for doc_id in range(train_size):
        words = tokenize_sentences[doc_id][order]
        for word in words:
            word_id = word_id_map[word]
            doc_word_str = str(doc_id) + ',' + str(word_id)
            if doc_word_str in doc_word_freq:
                doc_word_freq[doc_word_str] += 1
            else:
                doc_word_freq[doc_word_str] = 1
    doc_emb = []
    


    for i in range(train_size+test_size):
        words = tokenize_sentences[i][order]
        text = "".join(tokenize_sentences[i][0])

        if need_init:
            doc_emb.append(bert_encode(bert,text).cpu().numpy())
        doc_word_set = set()
        for word in words:
            if word in doc_word_set:
                continue
            j = word_id_map[word]
            key = str(i) + ',' + str(j)
            freq = doc_word_freq[key]
            row.append(i)
            col.append(train_size +test_size+ j)

            # col.append(train_size + j)
            idf = log(1.0 * train_size / (word_doc_freq[word_list[j]]+1))
            # w = freq * idf
            w = idf
            weight.append(w)
            doc_word_set.add(word)
            # print(doc_emb.shape,i,j)
            # doc_emb[i][j] = w/len(words) 
    return weight,row,col,word_doc_freq,node_size,doc_emb
def get_adj(tokenize_sentences,labels, train_size,test_size,word_id_map,word_list,lab_id_map,lab_list,bert,args):
    
    weight,row,col,word_doc_freq,node_size,doc_emb = get_weight(word_list,word_id_map,labels,train_size,test_size,tokenize_sentences,args,bert,order=0,need_window=True)
    weight_lab,row_lab,col_lab,lab_doc_freq,lab_size,_ = get_weight(lab_list,lab_id_map,labels,train_size,test_size,tokenize_sentences,args,bert,order=1,need_window=False,need_init=False)
    

  
            
            

    doc_emb = np.concatenate(doc_emb)        
    adj = sp.csr_matrix((weight, (row, col)), shape=(node_size, node_size))

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)    
    adj_lab = sp.csr_matrix((weight_lab, (row_lab, col_lab)), shape=(lab_size, lab_size))

    # build symmetric adjacency matrix
    adj_lab = adj_lab + adj_lab.T.multiply(adj_lab.T > adj_lab) - adj_lab.multiply(adj_lab.T > adj_lab)    
    return adj, adj_lab,doc_emb, word_doc_freq,lab_doc_freq
