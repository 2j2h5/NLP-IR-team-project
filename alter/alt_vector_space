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

import math

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
                naive_data += e
                for v in filtered:
                    tmp_data.append(v)
  
    data_index[idx]=naive_data
    parsed_data[idx]=tmp_data
    global sz
    sz=idx


inverted_index = {}
sv_count = {}
idf = {}

def make_inverted_index():
    for i in range(1,sz+1):
        sv_count[i] = Counter(parsed_data[i])
        for j in sv_count[i]:
            if j not in inverted_index:
                inverted_index[j] = set()
            inverted_index[j].add(i)


def make_idf():
    for t,v in inverted_index.items():
        idf[t]=math.log(sz/len(v))


def make_vector():
    global doc_vectors
    doc_vectors = {i:{} for i in range(1,sz+1)}
    global doc_norm
    doc_norm = {i:0.0 for i in range(1,sz+1)}
    for t,v in inverted_index.items():
        for i in v:
            tf = math.log(1+sv_count[i][t])
            doc_vectors[i][t] = tf * idf[t]
            doc_norm[i] += doc_vectors[i][t] ** 2
    
    for i in range(1,sz+1):
        doc_norm[i] = math.sqrt(doc_norm[i])

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
        elif e.startswith(".T") or e.startswith(".A") or e.startswith(".B") or e.startswith(".B"): p = "not_use"
        elif e.startswith(".W"): p = "word"
        else:
            if q_idx!=idx:
                continue
            u = word_tokenize(e.lower())
            filtered = [ps.stem(v) for v in u if v not in stop_words and v.isalnum()]  
            if p=="word":
                for v in filtered:
                    tmp_data.append(v)

        if(q_idx==idx) : naive_data += e


    qry_count = Counter(tmp_data)
    qry_vector = {}
    qry_norm = 0
    for t in qry_count:
        if t not in inverted_index: continue
        tf=math.log(1+qry_count[t])
        qry_vector[t]=tf*idf[t]
        qry_norm += qry_vector[t] ** 2
    
    qry_norm = math.sqrt(qry_norm)

    cos_value = []

    for i in range (1,sz+1):
        data_norm = doc_norm[i]

        q = 0
        if qry_norm==0 or data_norm==0: q=0
        else :
            for t in qry_vector:
                if t in doc_vectors[i]:
                    q += doc_vectors[i][t] * qry_vector[t]
            q /= qry_norm * data_norm
        
        cos_value.append((q,i))

    cos_value.sort(key=lambda x: -x[0])
    res = set()
    for i in range(len(rel_data[q_idx])):
        res.add(cos_value[i][1])
    
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
    make_idf()
    make_vector()
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