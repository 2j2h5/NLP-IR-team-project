import re

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
        tokens = re.findall(r'\b\w+\b',i.lower())
        for tk in tokens :
            if p == "title" : title[idx].append(tk)
            elif p == "author" : author[idx].append(tk)
            elif p == "word" : author[idx].append(tk)
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

