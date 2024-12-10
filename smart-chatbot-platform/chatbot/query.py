import json, numpy as np
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm
model = SentenceTransformer('all-MiniLM-L6-v2')
with open('index.json') as f: index=json.load(f)
def query(q):
    q_emb = model.encode(q)
    best=None; best_score=-1
    for k,v in index.items():
        s = np.dot(q_emb, np.array(v)) / (norm(q_emb)*norm(v))
        if s>best_score: best_score=s; best=k
    return best, best_score
if __name__=='__main__':
    q=input('Question: ')
    print(query(q))