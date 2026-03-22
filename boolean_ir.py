import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

sw = set(stopwords.words('english'))
ps = PorterStemmer()

# all 전처리 
all_path = "NLP-IR-team-project/data/CISI.ALL"

with open(all_path,'r',encoding='utf-8') as f:
    cisi_all = f.readlines()

n=0
for i in cisi_all:
    if i.startswith(".I") : n += 1

n+=1

title = [[] for _ in range(n)]
author = [[] for _ in range(n)]
word = [[] for _ in range(n)]
not_use = [[] for _ in range(n)]

p=""
idx = 0

for i in cisi_all:
    if i.startswith(".I") : idx+=1
    elif i.startswith(".T"): p = "title"
    elif i.startswith(".A"): p = "author"
    elif i.startswith(".W"): p = "word"
    elif i.startswith(".X"): p = "not_use"
    else :
        tmp = word_tokenize(i.lower())
        tokens = [ps.stem(u) for u in tmp if u not in sw and u.isalnum()]

        for tk in tokens :
            if p == "title" : title[idx].append(tk)
            elif p == "author" : author[idx].append(tk)
            elif p == "word" : word[idx].append(tk)
            else : not_use[idx].append(tk)

all_invert = {}

for i in range(1,n) : 
    for j in title[i] : 
        if j not in all_invert: 
            all_invert[j] = set()
        all_invert[j].add(i)
    for j in author[i] : 
        if j not in all_invert: 
            all_invert[j] = set()
        all_invert[j].add(i)
    for j in word[i] : 
        if j not in all_invert: 
            all_invert[j] = set()
        all_invert[j].add(i)



#qry 전처리
qry_path = "NLP-IR-team-project/data/CISI.QRY"

with open(qry_path,'r',encoding='utf-8') as f:
    cisi_qry = f.readlines()

n=0
for i in cisi_all:
    if i.startswith(".I") : n += 1
n+=1

query = [[] for _ in range(n)]

p=""
idx = 0
for i in cisi_qry:
    if i.startswith(".I") : idx+=1
    elif i.startswith(".W"): p = "word"
    elif i.startswith(".A"): p = "author"
    elif i.startswith(".W"): p = "word"
    elif i.startswith(".X"): p = "not_use"
    else :
        tmp = word_tokenize(i.lower())
        tokens = [ps.stem(u) for u in tmp if u not in sw and u.isalnum()]
        for tk in tokens : 
            if p== "word" :
                query[idx].append(tk)

#쿼리 번호 입력하면 and 연산으로 보여줌
a = int(input(":"))
sets = [all_invert.get(k) for k in query[a] if k in all_invert]
res1 = set.intersection(*sets)
res2 = set.union(*sets)

print("result(and) : ",list(res1))
#print("result(or) : ",list(res2))