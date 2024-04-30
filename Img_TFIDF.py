import json
import nltk
import math
#nltk.download('all')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

image_file = open('image_search_db.txt')
img_dict ={}

for data in image_file:
  current_line = data.split(":::")
  img_dict[current_line[0].strip()] = current_line[1].strip("\n")
#lowercase the words
img_lower = {}
for key, value in img_dict.items():
  img_lower[key] = value.lower()

#tokenize the data
def tokenize(param2):
  data_token = {}
  count = 0
  for key, value in param2.items():
    data_token[key] = word_tokenize(value)
  return data_token

#lemmitizing the words 
def lemmitization(param5):
  lemmi_dict_token = {}
  for key, value in param5.items():
    lis = []
    for data in value:
      lis.append(WordNetLemmatizer().lemmatize(data))
    lemmi_dict_token[key] = lis
  return lemmi_dict_token

#removing stop words
def remove_stopWords(param6):
  less_wordy_dict = {}
  stop_words = set(stopwords.words('english'))
 
  for key, value in param6.items():
    l = []
    for data in value:
      if data.isalpha():
        if data not in stop_words and len(data)>1:
          l.append(data)
    less_wordy_dict[key] = l
  return less_wordy_dict

########
img_token = tokenize(img_lower)
lemmitized_img_token = lemmitization(img_token)
noiseless_img_token = remove_stopWords(lemmitized_img_token)
#print(noiseless_img_token)
########

#checking if key is already present in inverted index
def check_key_present(key, data):
  for i in range(len(inverted_index[data])):
    if(key != inverted_index[data][i] ):
      continue
    else:
      return True
  return False

#creating boolean retrieval inverted index 
inverted_index = {}
def Create_inverted_index():
  song_num = 1
  for key, value in noiseless_img_token.items():
    for data in value:
      if data in inverted_index:
        if(check_key_present(song_num, data)):
          continue
        else:
          inverted_index[data].append(song_num)
      else:
        inverted_index[data] = [song_num]
    song_num += 1
  return inverted_index

#getting term frequency of each term in each doc
abs_term_freq = {}
def term_freq(img_desc_data):
  term_freq = {}
  for key, token in img_desc_data.items():
    token_freq = {}
    for data in token:
      if data in token_freq:
        token_freq[data] += 1
      else: 
        token_freq[data] = 1

    abs_term_freq[key] = token_freq
    l={}
    for k, v in token_freq.items():
      v1 = 1 + math.log10(v)
      l[k] = v1
    term_freq[key] = l
  return term_freq

#getting inverted doc frequency of each term in doc
def inv_doc_freq(invert_ind):
  idf_dict = {}
  num = len(img_dict)
  for key, value in invert_ind.items():
    deno = len(value)
    a = num/deno
    b = math.log10(a)
    idf_dict[key] = b
  return idf_dict

#getting tfidf of each term in each doc
def cal_tfidf(tf, idf):
  tfidf_dict = {}
  for key, value in tf.items():
    l = {}
    for k, v in value.items():
      val = v * idf[k]
      l[k] = val  
    tfidf_dict[key] = l
  return tfidf_dict

######
inverted_ind = Create_inverted_index()
img_desc_TF = term_freq(noiseless_img_token)
img_desc_IDF = inv_doc_freq(inverted_ind)
img_desc_TFIDF = cal_tfidf(img_desc_TF, img_desc_IDF)
######