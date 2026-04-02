import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

import os
import kagglehub
path = kagglehub.dataset_download("dmaso01dsta/cisi-a-dataset-for-information-retrieval")

from collections import Counter

data_index = {}
parsed_data = {}
sz = 0
def pre_all():
    file_path = os.path.join(path,'CISI.ALL')
    with open(file_path,'r') as f:
        lines = f.readlines()
    
    idx=0
    p=""
    naive_data=""
    tmp_data = []
    for e in lines:
        if e.startswith(".I") :
            if idx>0:
                data_index[idx]=naive_data
                parsed_data[idx]=tmp_data
                tmp_data=[] 
                naive_data="" 
            idx+=1
        elif e.startswith(".T"): p = "title"
        elif e.startswith(".A"): p = "author"
        elif e.startswith(".W"): p = "word"
        elif e.startswith(".X"): p = "not_use"
        else:
            u = word_tokenize(e.lower())
            filtered = [ps.stem(v) for v in u if v not in stop_words and v.isalnum()]
            
            if p=="title" or p=="word":
                #naive_data += e
                for v in filtered:
                    tmp_data.append(v)
  
    data_index[idx]=naive_data
    parsed_data[idx]=tmp_data
    global sz
    sz=idx


inverted_index = {}
sv_count = {}
df = {}
def make_inverted_index():
    for i in range(1,sz+1):
        sv_count[i] = Counter(parsed_data[i])
        for j in sv_count[i]:
            if j not in inverted_index:
                inverted_index[j] = set()
            inverted_index[j].add(i)


def make_df():
    for t,v in inverted_index.items():
        df[t]=len(v)


rel_data={}
def rel():
    file_path = os.path.join(path,'CISI.REL')
    with open(file_path,'r') as f:
        for i in f:
            j = i.split()
            
            q_id = int(j[0])
            doc_id = int(j[1])
            if q_id not in rel_data:
                rel_data[q_id]=set()
            rel_data[q_id].add(doc_id)


def do_query(q_idx):
    file_path = os.path.join(path,'CISI.QRY')
    with open(file_path,'r') as f:
        lines = f.readlines()

    tmp_data = []
    naive_data = ""
    idx=0
    for e in lines:
        if e.startswith(".I") :
            idx+=1
        elif e.startswith(".A") or e.startswith(".B") or e.startswith(".X"): p = "not_use"
        elif e.startswith(".W") or e.startswith(".T"): p = "word"
        else:
            if q_idx!=idx:
                continue
            u = word_tokenize(e.lower())
            filtered = [ps.stem(v) for v in u if v not in stop_words and v.isalnum()]  
            if p=="word":
                for v in filtered:
                    if v in inverted_index: tmp_data.append(v)

        #if(q_idx==idx) : naive_data += e

    tmp_data.sort(key=lambda x: df[x])  
    l=0
    r = len(tmp_data)-1
    v = 0 

    while l<=r:
        mid = (l+r)//2
        res = set()
        g=0
        for i in range(mid+1):
            if g==0:
                res = inverted_index[tmp_data[i]];
            else:
                res = res.intersection(inverted_index[tmp_data[i]])
            g += 1
        
        if len(res)>=len(rel_data[q_idx]):
            v=mid
            l=mid+1
        else: r=mid-1

    g=0
    res = set()
    for i in range(v+1):
        if g==0:
            res = inverted_index[tmp_data[i]];
        else:
            res = res.intersection(inverted_index[tmp_data[i]])
        g += 1
    
    tp = len(res.intersection(rel_data[q_idx]))
    fn = len(rel_data[q_idx]) - tp
    fp = len(res) - tp

    precision = tp / (tp+fp)
    recall = tp / (tp+fn)

    print("query id:",q_idx)
    print("precision:",precision,"recall:",recall)
    return precision,recall

def main():
    pre_all()
    make_inverted_index()
    make_df()
    rel()

    precision = 0 
    recall = 0
    for i in rel_data:
        p,r = do_query(i)
        precision += p
        recall += r
    
    print("averge precision:",precision/len(rel_data))
    print("averge_recall:",recall/len(rel_data))

if __name__ == '__main__':
    main()