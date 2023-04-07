import pandas as pd
def load_tsv(filename):
    labels = []
    sentences = []
    ## select the vare disease related to text
    labels_sub = [5,6,7,11,13,15,16,17,30,42,51,52,57,62]
    use_all = False
    with open(filename,'r',encoding='utf-8') as f:
        for line in f.readlines():

            label,word,select_word = line.split('\t')
            select_word = select_word.strip()
            label_int = list(map(int,label))
            if use_all:
                new_label = label_int
            else:
                new_label = [label_int[i] for i in labels_sub]
            labels.append(new_label)
            sentences.append((word,select_word))

    return sentences,labels
def samples(sentences,labels,rate):
    if rate > 1 or rate < 0:
        rate = 1
    leng = len(sentences)
    sample_size = int(leng*rate)
    return sentences[:sample_size],labels[:sample_size]
def get_dataset(args):
    train_sentences,train_labels = load_tsv('/home/wushuang/Sysu/ipynb/code/codeTextClaassification/vare_train_data_lab.tsv')
    val_sentences,val_labels = load_tsv('/home/wushuang/Sysu/ipynb/code/codeTextClaassification/vare_test_data_lab.tsv')
    test_sentences,test_labels = load_tsv('/home/wushuang/Sysu/ipynb/code/codeTextClaassification/vare_test_data_lab.tsv')
    train_sentences,train_labels = samples(train_sentences,train_labels,args.train_size)
    val_sentences,val_labels = samples(val_sentences,val_labels,args.train_size)
    test_sentences,test_labels = samples(test_sentences,test_labels,args.test_size)
    train_size = len(train_sentences)
    test_size = len(test_sentences)
    val_size = len(val_sentences)
    return train_sentences+val_sentences+test_sentences, train_labels+val_labels+test_labels, train_size,val_size, test_size
if __name__=="__main__":
    get_dataset(1)
