# Author: Harsh Kohli
# Date created: 25/09/20


from sentence_transformers import SentenceTransformer
import faiss
import yaml
import os
import numpy as np

config = yaml.safe_load(open('config.yml', 'r'))
os.environ["TORCH_HOME"] = config['base_model_dir']
model = SentenceTransformer(config['save_dir'])

index = faiss.read_index(config['index_file'])
test_file = open(config['test_file'], 'r', encoding='utf8')
k = config['num_searches']

category_file = open(config['category_map'], 'r', encoding='utf8')
topic_to_index_map = {}
for line in category_file.readlines():
    info = line.strip().split('\t')
    if len(info) > 1:
        topic_to_index_map[info[1]] = int(info[0])

query_topics_map = {}
for line in test_file.readlines():
    info = line.strip().split(config['delimiter'])
    query, topic = info[0], info[1]
    if topic in topic_to_index_map:
        if query in query_topics_map:
            query_topics_map[query].append((topic, topic_to_index_map[topic]))
        else:
            query_topics_map[query] = [(topic, topic_to_index_map[topic])]

count = 0
mrr = 0.0
for query, topics in query_topics_map.items():
    query_embedding = model.encode(query)
    top_k = index.search(np.asarray([query_embedding]), k)
    result_indices = list(top_k[1][0])
    for topic, index in topics:
        if index in result_indices:
            rr = 1.0 / result_indices.index(index)
            mrr = mrr + rr
            count = count + 1

print('MRR is ' + str(mrr))
