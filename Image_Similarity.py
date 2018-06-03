
# coding: utf-8

import pandas as pd
df = pd.read_csv('input.csv')

df.columns = ("label", "features")
df.shape


pdf = pd.eval(df['features'])
from scipy.stats import pearsonr

ps = pearsonr(pdf[7], pdf[28])
print(ps)

from sklearn.metrics.pairwise import cosine_similarity
df = df.eval(df.features)
cs = cosine_similarity(df)
cs.shape
numpy.asmatrix(cs[0])
#cs[7]

type(cs)
import numpy 
ncs = numpy.sort(cs[7])
ncs[::-1]


# Faiss (Facebook AI Research) Implementation for Similar Image Search

import numpy
import faiss

xb = pd.eval(df['features'])
xq = pd.eval(df['features'])
xb = numpy.asarray(xb, dtype=numpy.float32)
xq = numpy.asarray(xq, dtype=numpy.float32)
#xb.shape

   
index = faiss.IndexFlatL2(100)   

index.add(xb)                  
print(index.ntotal)

k = 5             
D, I = index.search(xq, k)     
print (I)

