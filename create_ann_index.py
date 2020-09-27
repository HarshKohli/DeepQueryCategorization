# Author: Harsh Kohli
# Date created: 08/09/20

import os
import yaml
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

config = yaml.safe_load(open('config.yml', 'r'))

train_file = open(config['train_file'], 'r', encoding='utf8')
batch_size = config['batch_size']

model = SentenceTransformer(config['save_dir'])

classes = []

print('Creating class index...')
for line in train_file.readlines():
    category = line.strip().split(config['delimiter'])[1]
    if category not in classes:
        classes.append(category)

train_file.close()
batches = [classes[i * batch_size:(i + 1) * batch_size] for i in range((len(classes) + batch_size - 1) // batch_size)]

ann_file = config['index_file']
if os.path.exists(ann_file):
    os.remove(ann_file)
    index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
else:
    index = faiss.IndexIDMap(faiss.IndexFlatIP(768))

category_map = open(config['category_map'], 'w', encoding='utf8')
class_num = 0
for batch in batches:
    sentence_embeddings = model.encode(batch)
    ids = range(class_num, class_num + len(batch))
    class_num = class_num + len(batch)
    index.add_with_ids(np.asarray(sentence_embeddings), np.asarray(ids))
    for id, sentence in zip(ids, batch):
        category_map.write(str(id) + '\t' + str(sentence) + '\n')

faiss.write_index(index, ann_file)
print('Done creating index!')
