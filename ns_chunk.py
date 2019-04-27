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
  #print(text)
  tokenized_s1 = tokenizer.tokenize(sentence1)
  tokenized_text = tokenizer.tokenize(text)
  indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
  segments_ids = [1]*len(tokenized_text)
  for x in range(len(tokenized_s1)):
    segments_ids[x] = 0
  tokens_tensor = torch.tensor([indexed_tokens])
  segments_tensors = torch.tensor([segments_ids])
  predictions = model(tokens_tensor, segments_tensors)
  #print(predictions)
  #print(predictions[0, 0])
  #print(predictions[0, 1])
  res = torch.nn.functional.softmax(predictions[0].data, dim = 0)
  fin = res.tolist()
  #res = predictions[0].data.tolist()
  #fin = res
  #odds = math.exp(-1*res[0])
  #fin[0] = 1/(odds + 1)
  #fin[1] = 1 - fin[0]
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

#s1 = "What is the effect of online availability of journal issues?"
#s2 = "I will show, however, that even as deeper journal back issues became available online, scientists and scholars cited more recent articles; even as more total journals became available online, fewer were cited."
#print(sentence_pred(s1, s2))
#p = "What is the effect of online availability of journal issues? It is possible that by making more research more available, online searching could conceivably broaden the work cited and lead researchers, as a collective, away from the “core” journals of their fields and to dispersed but individually relevant work. I will show, however, that even as deeper journal back issues became available online, scientists and scholars cited more recent articles; even as more total journals became available online, fewer were cited."
#p = "None of the women, though, gave them any indication that Castro's two older brothers, who've been in custody since Monday, were involved, Tomba said. Prosecutors brought no charges against the brothers, citing a lack of evidence."
#p = "The White House also is seeking to push through other rules that would tighten student and investor visas, and it is pursuing what some describe as its most ambitious goal on immigration: preventing immigrants from coming or becoming citizens if they are likely to use publicly funded benefits. Earlier this month, for instance, Google unveiled a high-profile, global, independent ethics council to guide it on the responsible development of all of its AI-related research and products. It is impossible to build a car without cerium, a smartphone without europium, a guided missile without neodymium."
#p = "Who was Jim Morrison ? Jim Morrison was a puppeteer"
#p = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
#p = "How old are you ? The Eiffel Tower is in Paris"
#p = "[CLS] How old are you ? [SEP] The Eiffel Tower is in Paris [SEP]"
#p = "What is the effect of online availability of journal issues? It is possible that by making more research more available, online searching could conceivably broaden the work cited and lead researchers, as a collective, away from the “core” journals of their fields and to dispersed but individually relevant work. I will show, however, that even as deeper journal back issues became available online, scientists and scholars cited more recent articles; even as more total journals became available online, fewer were cited."
#print(paragraph_score(p))

#data_df = pd.read_csv('data/startribune_paragraph.csv')
#data_df['predicted scores'] = data_df['text'].apply(lambda x: paragraph_score(x))
#data_df.to_csv('data/paragraph_outputs.csv')

fp = open("data/1977_458.txt")
data = fp.read()

from itertools import islice
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def chunked(iterable, n):
    chunksize = int(math.ceil(len(iterable) / n))
    print(chunksize)
    return (iterable[i * chunksize:i * chunksize + chunksize]
            for i in range(n))

arr = tokenize.sent_tokenize(data)
res = list(chunks(arr,20))

fin = []
for item in res:
  l = " ".join(item)
  score = paragraph_score(l)
  fin.append(score)
  print(score)

print(fin)

import simplejson
f = open('data/sw.txt', 'w')
simplejson.dump(fin, f)
f.close()
