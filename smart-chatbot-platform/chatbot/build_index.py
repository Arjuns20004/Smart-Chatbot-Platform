from sentence_transformers import SentenceTransformer
import os, json
model = SentenceTransformer('all-MiniLM-L6-v2')
docs = {}
for fn in os.listdir('docs'):
    with open(os.path.join('docs',fn)) as f: docs[fn]=f.read()
embeddings = {k: model.encode(v).tolist() for k,v in docs.items()}
with open('index.json','w') as f: json.dump(embeddings,f)
print('Index built')