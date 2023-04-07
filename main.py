from build_dataset import get_dataset
from preprocess import encode_labels,preprocess_data
from build_graph import get_adj
from train import train_model
from utils import *
from model import GCN
from evaluate import get_weights_hidden, get_test_emb, test_model
import argparse
import torch
import torch.optim as optim
import scipy.sparse as sp
import torch.nn as nn
from sklearn.metrics import classification_report
import json
import pickle as pkl
from utils import cal_accuracy
import torch
import torch.nn as nn

# use bert to classifify the text
class BertCLS(nn.Module):
    def __init__(self,bert_embeding,bert_encoder,n_class):
        super(BertCLS,self).__init__()

        self.bert_embeding = bert_embeding
        self.bert_encoder = bert_encoder
        self.encoder_l = len(bert_encoder)     
        kernal_number = 16
        self.conv = nn.ModuleList(
            [nn.Conv2d(1,kernal_number,kernel_size=(i,768),stride=1) for i in [4,5,6]]
        )
        self.sigmoid = nn.Sigmoid()
        self.ff = nn.Linear(kernal_number*3,n_class)

    def encode(self,x,m):
        x = self.bert_embeding(x,m)
        for i in range(self.encoder_l):
                x = self.bert_encoder[i](x)[0]
        y_ = []
        for i in range(3):
                y_.append(self.conv[i](x.unsqueeze(1)).squeeze(1).max(-2).values.squeeze(-1))
            
        y_ = torch.cat(y_,-1)
        return y_
    def forward(self,x,m):
        x = self.bert_embeding(x,m)
        for i in range(self.encoder_l):
                x = self.bert_encoder[i](x)[0]
        y_ = []
        for i in range(3):
                y_.append(self.conv[i](x.unsqueeze(1)).squeeze(1).max(-2).values.squeeze(-1))
            
        y_ = torch.cat(y_,-1)
        y_ = nn.functional.relu(y_)
        y_ = self.ff(y_)
        return self.sigmoid(y_)
### tool to save the json file
def save_json(filename,obj):
    with open(filename,'w',encoding='utf-8') as f:
        return json.dump(obj,f,indent=4,ensure_ascii=False)

def load_vocab(filename):
    vocab_list = []
    with open(filename,'r',encoding='utf-8') as f:
        for line in f.readlines():
            vocab_list.append(line.strip())
    return vocab_list
def load_json(filename):
    with open(filename,'r',encoding='utf-8') as f:
        return json.load(f)
def save_vocab(vocab_list,filename):
    with open(filename,'w',encoding='utf-8') as f:
        print("\n".join(vocab_list),file=f)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='R8', help='Dataset string: R8, R52, OH, 20NGnew, MR')
parser.add_argument('--train_size',  type=float, default=1, help='If it is larger than 1, it means the number of training samples. If it is from 0 to 1, it means the proportion of the original training set.')
parser.add_argument('--test_size',  type=float, default=1, help='If it is larger than 1, it means the number of training samples. If it is from 0 to 1, it means the proportion of the original training set.')
parser.add_argument('--remove_limit', type=int, default=10, help='Remove the words showing fewer than 2 times')
parser.add_argument('--use_gpu', type=int, default=1, help='Whether to use GPU, 1 means True and 0 means False. If True and no GPU available, will use CPU instead.')
parser.add_argument('--shuffle_seed',type = int, default = None, help="If not specified, train/val is shuffled differently in each experiment")
parser.add_argument('--hidden_dim',type = int, default = 48, help="The hidden dimension of GCN model")
parser.add_argument('--dropout',type = float, default = 0.5, help="The dropout rate of GCN model")
parser.add_argument('--learning_rate',type = float, default = 8e-3, help="Learning rate")
parser.add_argument('--weight_decay',type = float, default = 0, help="Weight decay, normally it is 0")
parser.add_argument('--early_stopping',type = int, default = 10, help="Number of epochs of early stopping.")
parser.add_argument('--epochs',type = int, default = 4000, help="Number of maximum epochs")
parser.add_argument('--multiple_times',type = int, default = 5, help="Running multiple experiments, each time the train/val split is different")
parser.add_argument('--easy_copy',type = int, default = 1, help="For easy copy of the experiment results. 1 means True and 0 means False.")
parser.add_argument('--use_emb',type = int, default = 0, help="if use the pre training embdeing")
parser.add_argument('--batch_size',type = int, default = 32, help="if use the pre training embdeing")
parser.add_argument('--lambda1',type = float, default = 1.0, help="if use the pre training embdeing")
parser.add_argument('--use_load',type = bool, default = True, help="if use the pre training embdeing")
args = parser.parse_args()

device = decide_device(args)

# Get dataset
sentences, labels, train_size, val_size,test_size = get_dataset(args)
train_size += val_size
train_sentences = sentences[:train_size]
test_sentences = sentences[train_size:]
train_labels = labels[:train_size]
test_labels = labels[train_size:]

# Preprocess text and labels

use_load = True
num_class = len(labels[0])
print(num_class)
tokenize_sentences, word_list,lab_list = preprocess_data(train_sentences, test_sentences, args)

load_dataset = "load_dataset_ehr"
#when the use_load is True, we can load the word embedding from the save file, or we should calculate them by pre train model
if not use_load:
    bert = torch.load('/home/wushuang/Sysu/multi_label/bert/model_name.pth')
    save_vocab(word_list,f'/home/wushuang/Sysu/ipynb/code/codeTextClaassification/InductGCN_basebert/{load_dataset}/word_list.txt')
    save_vocab(lab_list,f'/home/wushuang/Sysu/ipynb/code/codeTextClaassification/InductGCN_basebert/{load_dataset}/lab_list.txt')
    vocab_length = len(word_list)
    word_id_map = {}
    for i in range(vocab_length):
        word_id_map[word_list[i]] = i
    lab_length = len(lab_list)
    lab_id_map = {}
    for i in range(lab_length):
        lab_id_map[lab_list[i]] = i

    print("There are", vocab_length,lab_length, "unique words in total.")   

    # Generate Graph
    adj, adj_lab,doc_emb, word_doc_freq,lab_doc_freq = get_adj(tokenize_sentences,labels,train_size,test_size,word_id_map,word_list,lab_id_map,lab_list,bert,args)
    doc_emb = doc_emb.reshape(-1,48)
    save_json(f"/home/wushuang/Sysu/ipynb/code/codeTextClaassification/InductGCN_basebert/{load_dataset}/word_doc_freq.json",word_doc_freq)
    save_json(f"/home/wushuang/Sysu/ipynb/code/codeTextClaassification/InductGCN_basebert/{load_dataset}/lab_doc_freq.json",lab_doc_freq)
    with open(f"/home/wushuang/Sysu/ipynb/code/codeTextClaassification/InductGCN_basebert/{load_dataset}/ind.graph", 'wb') as f:
        pkl.dump(adj, f)
    with open(f"/home/wushuang/Sysu/ipynb/code/codeTextClaassification/InductGCN_basebert/{load_dataset}/ind.adj_lab", 'wb') as f:
        pkl.dump(adj_lab, f)
    with open(f"/home/wushuang/Sysu/ipynb/code/codeTextClaassification/InductGCN_basebert/{load_dataset}/ind.doc_emb", 'wb') as f:
        pkl.dump(doc_emb, f)

    adj, norm_item = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)
    adj_lab, norm_item_lab = normalize_adj(adj_lab + sp.eye(adj_lab.shape[0]))
    adj_lab = sparse_mx_to_torch_sparse_tensor(adj_lab).to(device)
    features = torch.FloatTensor(doc_emb).to(device)


    # Generate Test input
    test_emb, tokenized_test_edge,tokenized_test_edge_lab = get_test_emb(tokenize_sentences[train_size:], word_id_map, vocab_length, word_doc_freq, word_list,lab_id_map,lab_length,lab_doc_freq,lab_list, train_size, norm_item,norm_item_lab,bert)

    with open(f"/home/wushuang/Sysu/ipynb/code/codeTextClaassification/InductGCN_basebert/{load_dataset}/ind.test_emb", 'wb') as f:
        pkl.dump(test_emb, f)
    with open(f"/home/wushuang/Sysu/ipynb/code/codeTextClaassification/InductGCN_basebert/{load_dataset}/ind.tokenized_test_edge", 'wb') as f:
        pkl.dump(tokenized_test_edge, f)
else:
    bert = torch.load('/home/wushuang/Sysu/multi_label/bert/model_name.pth')
    word_list = load_vocab(f'/home/wushuang/Sysu/ipynb/code/codeTextClaassification/InductGCN_basebert/{load_dataset}/word_list.txt')
    lab_list = load_vocab(f'/home/wushuang/Sysu/ipynb/code/codeTextClaassification/InductGCN_basebert/{load_dataset}/lab_list.txt')
    vocab_length = len(word_list)
    lab_length = len(lab_list)
    word_id_map = {}
    for i in range(vocab_length):
        word_id_map[word_list[i]] = i
    lab_id_map = {}
    for i in range(lab_length):
        lab_id_map[lab_list[i]] = i
    if not args.easy_copy:
        print("There are", vocab_length,lab_list, "unique words in total.")   
    adj = pkl.load(open(f"/home/wushuang/Sysu/ipynb/code/codeTextClaassification/InductGCN_basebert/{load_dataset}/ind.graph", "rb"))
    adj_lab = pkl.load(open(f"/home/wushuang/Sysu/ipynb/code/codeTextClaassification/InductGCN_basebert/{load_dataset}/ind.adj_lab", "rb"))
    doc_emb = pkl.load(open(f"/home/wushuang/Sysu/ipynb/code/codeTextClaassification/InductGCN_basebert/{load_dataset}/ind.doc_emb", "rb"))
    doc_emb = doc_emb.reshape(-1,48)
    word_doc_freq = load_json(f"/home/wushuang/Sysu/ipynb/code/codeTextClaassification/InductGCN_basebert/{load_dataset}/word_doc_freq.json")
    lab_doc_freq = load_json(f"/home/wushuang/Sysu/ipynb/code/codeTextClaassification/InductGCN_basebert/{load_dataset}/lab_doc_freq.json")
    adj, norm_item = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)
    adj_lab, norm_item_lab = normalize_adj(adj_lab + sp.eye(adj_lab.shape[0]))
    adj_lab = sparse_mx_to_torch_sparse_tensor(adj_lab).to(device)
    features = torch.FloatTensor(doc_emb).to(device)
    
    test_emb, tokenized_test_edge,tokenized_test_edge_lab = get_test_emb(tokenize_sentences[train_size:], word_id_map, vocab_length, word_doc_freq, word_list,lab_id_map,lab_length,lab_doc_freq,lab_list, train_size, norm_item,norm_item_lab,bert)

labels = torch.FloatTensor(labels).to(device)    
label_truth = torch.eye(num_class,num_class).long().to(device)  

class_freq = (labels[:train_size].sum(axis=0).cpu().numpy()/train_size).tolist()
criterion = nn.MSELoss()



if args.multiple_times:
    test_acc_list = []
    for t in range(args.multiple_times):
        if not args.easy_copy:
            print("Round",t+1)
        model = GCN(nfeat1=vocab_length+num_class,nfeat2=lab_length+num_class, nhid=args.hidden_dim, nclass=num_class, dropout=args.dropout).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        
        idx_train, idx_val,idx_test = generate_train_val(args, train_size,val_size,test_size)
        train_model(args, model, optimizer,criterion, features, adj,adj_lab, labels,label_truth, idx_train, idx_val,idx_test,show_result=True)
        model = torch.load(f'model/best_model_{args.lambda1}.pth')
        model_weights_list = get_weights_hidden(model,features,adj,adj_lab,train_size,test_size,vocab_length,lab_length,idx_train+idx_val+idx_test,args.lambda1)
        test_result = test_model(model, test_emb, tokenized_test_edge,tokenized_test_edge_lab,model_weights_list,device,args.lambda1)
        test_acc_list.append(cal_accuracy(test_result,labels[train_size:].cpu()))
    if args.easy_copy:

        print("%.4f"%np.mean(test_acc_list), end = ' ± ')
        print("%.4f"%np.std(test_acc_list))

    else: 
        for t in test_acc_list:
            print("%.4f"%t)       
        print("Test Accuracy:",np.round(test_acc_list,4).tolist())
        print("Mean:%.4f"%np.mean(test_acc_list))
        print("Std:%.4f"%np.std(test_acc_list))
