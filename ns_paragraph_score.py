import torch
import math
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForNextSentencePrediction
from nltk import tokenize
import pandas as pd
import numpy as np

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
model.eval()

# create function to get the predicted weights for nextSent and notNext
def sentence_pred(sentence1, sentence2):
  #text = "[CLS] " + sentence1 + " [SEP] " + sentence2 + " [SEP]"
  sentence1 = "[CLS] " + sentence1
  sentence2 = "[SEP] " + sentence2 + " [SEP]"
  text = sentence1 + " " + sentence2
  tokenized_s1 = tokenizer.tokenize(sentence1)
  tokenized_text = tokenizer.tokenize(text)
  indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
  segments_ids = [1]*len(tokenized_text)
  for x in range(len(tokenized_s1)):
    segments_ids[x] = 0
  tokens_tensor = torch.tensor([indexed_tokens])
  segments_tensors = torch.tensor([segments_ids])
  predictions = model(tokens_tensor, segments_tensors)
  res = torch.nn.functional.softmax(predictions[0].data, dim = 0)
  fin = res.tolist()
  return fin

# function to get avg score of a paragraph
def paragraph_score(par):
  split_sent = tokenize.sent_tokenize(par)
  
  if len(split_sent) <= 1:
    return -1
  
  score = 0
  for i in range(len(split_sent)-1):
      score += sentence_pred(split_sent[i], split_sent[i+1])[0]
  return score/(len(split_sent)-1)

data_df = pd.read_csv('data/startribune_paragraph.csv')
first_article_df = data_df[data_df.article == 8165]
merged_df = pd.concat([first_article_df, first_article_df.shift(-1).add_prefix('next_')['next_text']], axis=1)
# convert types
merged_df["text"]= merged_df["text"].astype(str)

merged_df['last_sentence'] = merged_df['text'].apply(lambda x: tokenize.sent_tokenize(x)[-1])
merged_df['second_last_sentence'] = merged_df['text'].apply(lambda x: tokenize.sent_tokenize(x)[-2])
merged_df = merged_df[:-1]
merged_df["next_text"]= merged_df["next_text"].astype(str)
merged_df['first_sentence'] = merged_df['next_text'].apply(lambda x: tokenize.sent_tokenize(x)[0])
merged_df['second_sentence'] = merged_df['next_text'].apply(lambda x: tokenize.sent_tokenize(x)[1])
merged_df['last_first_score'] = merged_df[['last_sentence','first_sentence']].apply(lambda x: sentence_pred(*x), axis=1)
merged_df['sec_last_first_score'] = merged_df[['second_last_sentence','first_sentence']].apply(lambda x: sentence_pred(*x), axis=1)
merged_df['sec_last_second_score'] = merged_df[['second_last_sentence','second_sentence']].apply(lambda x: sentence_pred(*x), axis=1)
merged_df['last_second_score'] = merged_df[['last_sentence','second_sentence']].apply(lambda x: sentence_pred(*x), axis=1)
merged_df.to_csv('data/par_sent_combo_outputs.csv')
