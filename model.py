import math
import numpy as np
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
# import torch.nn.functional as F
     

class GraphConvolution(Module):

    def __init__(self, in_features, out_features,  feature_less=None,drop_out = 0, activation=None, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.zeros(1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters(in_features, out_features)
        self.dropout = torch.nn.Dropout(drop_out)
        self.activation =  activation
        if feature_less:
            self.weight_less = Parameter(torch.FloatTensor(in_features, out_features))

    def reset_parameters(self,in_features, out_features):
        stdv = np.sqrt(6.0/(in_features+out_features))
        self.weight.data.uniform_(-stdv, stdv)


    def forward(self, input, adj, feature_less = False):
        if feature_less:           
            support1 = input
            support2 = self.weight
            support = torch.cat([support1,support2],dim=0)
            support = self.dropout(support)
        else:    
            input = self.dropout(input)
            support = torch.spmm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
class GCN(nn.Module):
    def __init__(self, nfeat1,nfeat2, nhid, nclass, dropout):
        super(GCN, self).__init__()
        ## lab stand for the layer in laboratory space 
        self.gc1 = GraphConvolution(nfeat1, nhid, dropout, activation = nn.ReLU())
        self.gc2 = GraphConvolution(nhid, nclass*3, dropout)

        self.gc_lab1 = GraphConvolution(nfeat2, nhid, dropout, activation = nn.ReLU())
        self.gc_lab2 = GraphConvolution(nhid, nclass*3, dropout)

        self.att1 = nn.Linear(nhid,1)
        self.att2 = nn.Linear(nclass*3,1)
        self.fc = nn.Linear(nclass*3,nclass)
        self.nclass = nclass

    def encode(self, x, adj,adj_lab,idx,lambda1):

        x1 = self.gc1(x, adj, feature_less = True) 
        x1_lab = self.gc_lab1(x, adj_lab, feature_less = True) 


        x2 = self.gc2(x1,adj)
        x2_lab = self.gc_lab2(x1_lab, adj_lab) 
        x2_mix = x2[idx] + x2_lab[idx] * lambda1

        x2 = nn.functional.relu(x2_mix)
        x2 = self.fc(x2)
        x2 = torch.sigmoid(x2)
        return x1,x1_lab,x2[-self.nclass:]
    def forward(self, x, adj,adj_lab,idx,lambda1):

        x1 = self.gc1(x, adj, feature_less = True) 
        x1_lab = self.gc_lab1(x, adj_lab, feature_less = True) 


        x2 = self.gc2(x1,adj)
        x2_lab = self.gc_lab2(x1_lab, adj_lab) 
        x2_mix = x2[idx] + x2_lab[idx] * lambda1
        x2 = nn.functional.relu(x2_mix)
        x2 = self.fc(x2)
        x2 = torch.sigmoid(x2)
        return x1, x2 
           
