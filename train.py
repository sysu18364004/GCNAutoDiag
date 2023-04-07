import time
import numpy as np
from utils import cal_accuracy
from tqdm import tqdm
import torch
def adjust_learning_rate(optimizer, epoch, args):
            """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
            if epoch == 100:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1e-2
            if epoch == 1000:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 2e-3
def train_model(args, model, optimizer,criterion, features, adj,adj_lab, labels,label_truth, idx_train, idx_val, idx_test,show_result = True):
    val_loss = []

    save_path = f'model/best_model_{args.lambda1}.pth'
    best_val_f1 = 0
    idx_train_out = idx_train.copy()
    idx_label_out = idx_train.copy()
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        _, output= model(features, adj,adj_lab,idx_train+ idx_val+idx_test,args.lambda1)
        lambdas = 0.5

        train_output = output[idx_train_out]
        train_label = labels[idx_label_out]
   
        mask = train_label > 0.5
        loss_train = criterion(train_output*mask, train_label*mask) + lambdas*criterion(train_output, train_label)
        acc_train = cal_accuracy(output[idx_train], labels[idx_train])

        # the number of label nodes is much smaller than medical record, so we use the little value 0.000001.
        loss =  loss_train + 0.000001*criterion(output[-label_truth.shape[0]:], label_truth.float())
        loss.backward()
        optimizer.step()
        
        
        model.eval()
        _, output =  model(features, adj,adj_lab,idx_train+ idx_val+idx_test,args.lambda1)
        mask = labels[idx_val] > 0.5
        loss_val = criterion(output[idx_val]*mask, labels[idx_val]*mask)+lambdas*criterion(output[idx_val], labels[idx_val])
        loss = criterion(output[idx_val], labels[idx_val])
        val_loss.append(loss_val.item())



        acc_val = cal_accuracy(output[idx_val], labels[idx_val])
        if acc_val > best_val_f1:
            best_val_f1 = acc_val
            torch.save(model,save_path)
        acc_test = cal_accuracy(output[idx_test], labels[idx_test])
        if show_result:
            print(  'Epoch: {:04d}'.format(epoch+1),
                    'loss_train: {:.4f}'.format(loss_train.item()),
                    'f1_train: {:.4f}'.format(acc_train),
                    'loss_val: {:.4f}'.format(loss_val.item()),
                    'f1_val: {:.4f}'.format(acc_val),
                    'f1_test: {:.4f}'.format(acc_test)
            )
