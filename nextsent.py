import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForNextSentencePrediction

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenized input
#sentence1 = "Who was Jim Morrison ?"
#sentence2 = "Jim Morrison was a puppeteer"
sentence1 = "What is the effect of online availability of journal issues?"
#sentence2 = "It is possible that by making more research more available, online searching could conceivably broaden the work cited and lead researchers, as a collective, away from the “core” journals of their fields and to dispersed but individually relevant work."
sentence2 = "I will show, however, that even as deeper journal back issues became available online, scientists and scholars cited more recent articles; even as more total journals became available online, fewer were cited."
text = sentence1 + " " + sentence2
tokenized_s1 = tokenizer.tokenize(sentence1)

#text = "Who was Jim Morrison ? Jim Morrison was a puppeteer"
tokenized_text = tokenizer.tokenize(text)
print(tokenized_text)
# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
print(indexed_tokens)
# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
segments_ids = [1]*len(tokenized_text)

for x in range(len(tokenized_s1)):
  segments_ids[x] = 0

#segments_ids = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

# Load pre-trained model (weights)
model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
model.eval()

# Predict is Next Sentence ?
predictions = model(tokens_tensor, segments_tensors)
print(predictions[0].data.tolist())
