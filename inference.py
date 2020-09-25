# Author: Harsh Kohli
# Date created: 25/09/20

from torch.utils.data import DataLoader
from sentence_transformers.readers import InputExample
from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses, evaluation
import logging
import yaml
import os

config = yaml.safe_load(open('config.yml', 'r'))
os.environ["TORCH_HOME"] = config['base_model_dir']

model = SentenceTransformer(config['save_dir'])

