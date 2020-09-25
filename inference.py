# Author: Harsh Kohli
# Date created: 25/09/20


from sentence_transformers import SentenceTransformer
import faiss
import yaml
import os

config = yaml.safe_load(open('config.yml', 'r'))
os.environ["TORCH_HOME"] = config['base_model_dir']
model = SentenceTransformer(config['save_dir'])

test_file = open(config['test_file'], 'r', encoding='utf8')

query_topics_map = {}
for line in test_file.readlines():
    info = line.split(config['delimiter'])
    query, topic = info[0], info[1]
    if query in query_topics_map:
        query_topics_map[query].append(topic)
    else:
        query_topics_map[query] = [topic]

for query, topics in query_topics_map.items():
    query_embedding = model.encode(query)
    print('here')

