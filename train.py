# # Author: Harsh Kohli
# # Date created: 09/08/20

from torch.utils.data import DataLoader
from sentence_transformers.readers import InputExample
from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses, evaluation
import logging
import yaml
import os

os.environ["TORCH_HOME"] = "model/base"
config = yaml.safe_load(open('config.yml', 'r'))

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

model = SentenceTransformer(config['base_model'])

train_samples = []
train_file = open(config['train_file'], 'r', encoding='utf8')

print('Reading train file...')
for index, line in enumerate(train_file.readlines()):
    if index % 10000 == 0:
        print('Processing row ' + str(index))
    info = line.split(config['delimiter'])
    try:
        train_samples.append(InputExample(texts=[info[0], info[1]], label=1))
        train_samples.append(InputExample(texts=[info[1], info[0]], label=1))
    except:
        continue
print('Done reading train file!')

dev_sentences1 = []
dev_sentences2 = []
dev_labels = []
dev_file = open(config['test_file'], 'r', encoding='utf8')

print('Reading dev file...')
for line in dev_file.readlines():
    info = line.split(config['delimiter'])
    try:
        dev_sentences2.append(info[1])
        dev_sentences1.append(info[0])
        dev_labels.append(1)
    except:
        continue
print('Done reading dev file!')

batch_size = config['batch_size']
train_dataset = SentencesDataset(train_samples, model=model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
train_loss = losses.MultipleNegativesRankingLoss(model)

model_save_path = config['save_dir']
os.makedirs(model_save_path, exist_ok=True)

evaluators = [evaluation.BinaryClassificationEvaluator(dev_sentences1, dev_sentences2, dev_labels)]
seq_evaluator = evaluation.SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[-1])
seq_evaluator(model, epoch=0, steps=0, output_path=model_save_path)

model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=seq_evaluator,
          epochs=config['num_epochs'],
          warmup_steps=config['warmup_steps'],
          output_path=model_save_path,
          output_path_ignore_not_empty=True
          )
