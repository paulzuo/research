#with open('data/1989_3628.txt', 'r') as file:
#    data = file.read().replace('\n', '')

fp = open("data/1995_3484.txt")
data = fp.read()

from itertools import islice
from nltk import tokenize
import pandas as pd
import numpy as np
import math

def chunk(it, size):
    it = iter(it)
    return iter(lambda: list(islice(it, size)), [])

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
def chunked(iterable, n):
    chunksize = int(math.ceil(len(iterable) / n))
    return (iterable[i * chunksize:i * chunksize + chunksize]
            for i in range(n))

arr = tokenize.sent_tokenize(data)
print(len(arr))
res = list(chunks(arr,20))
print(len(res))
#print(res)

fin = []
for item in res:
  fin.append(" ".join(item))

print(len(fin))
idx = list(range(1, len(fin)+1))
print(len(idx))
data = [idx, fin]
df = pd.DataFrame(
    {'chunk_number': idx,
     'paragraph': fin
    })
#df.columns = ['chunk_number', 'paragraph']
#print(fin)
df.to_csv('data/toy_story_paragraphs.csv')

