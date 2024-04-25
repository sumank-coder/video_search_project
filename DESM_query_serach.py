# your code here
import json
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import nltk
#nltk.download('all')
from nltk.tokenize import word_tokenize
import numpy as np
import Img_TFIDF
import query_search
np.random.seed(0)

#create the training set and tokenize the data
data_token = {}
train_set = []
data_token = Img_TFIDF.noiseless_img_token

for key,value in data_token.items():
  train_set.append(value)

#train the model
W2V_model = Word2Vec(train_set, min_count=1, vector_size=100, window=5, workers=4, epochs = 20)
vocab = list(W2V_model.wv.key_to_index.keys())

#create document embedding vector
def embedding_vector(doc_tokens):
  embeddings=[]
  if len(doc_tokens)<1:
    return np.zeros(100)
  else:
    for token in doc_tokens:
      if token in vocab:
        embeddings.append(W2V_model.wv.get_vector(token))
      else:
        embeddings.append(np.random.rand(100))
        
  # mean the vectors of individual words to get the vector of the document
  return np.mean(embeddings, axis=0)

from sklearn.metrics.pairwise import cosine_similarity

# create embedding vector for all the documents:
Doc_embedding_list = {}
for key, value in data_token.items():
  Doc_embedding_list[key]= embedding_vector(value)
final_result={}

def DESM_search(query):
  query_processed = query_search.pre_process_query(query)
  query_embedd = embedding_vector(query_processed)
  #get cosine similarity
  for key,value in Doc_embedding_list.items():
    final_result[key] = cosine_similarity([np.array(value)], [np.array(query_embedd)])
  #based on similarity sort the result
  final_result_sorted = dict(sorted(final_result.items(), key=lambda x:x[1], reverse= True))
  images = query_search.Top10_image_list(final_result_sorted)
  return images

  