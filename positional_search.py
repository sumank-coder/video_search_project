#adding positional index to support phrase queries
#Using (1)+(2) processed data

#creating positional index list
import Img_TFIDF
import query_search

positional_index = {}
data_token = Img_TFIDF.noiseless_img_token
for key, value in data_token.items():
  d = {}
  i = 0
  for data in value: 
    i += 1
    if data in d:
      d[data].append(i)
    else: 
      d[data] = [i]

  positional_index[key] = d

#query search engine
def Pos_query_search(phrase):
  phrase_list = query_search.pre_process_query(phrase)
  positional_index_allquery = {}
  img_list_allquery = []

  for key, value in positional_index.items():
    l = []
    for query in phrase_list:
      if query in value: 
        l.append(value[query])

    if(len(phrase_list) == len(l)):
      query1 = l[0]
      n = len(l)

      for data in query1:
        count = 0 
        for i in range(1, n):
          for j in range(5): 
            if data+i+j in l[i]:
                count += 1
                break
        if(count == n-1):
          img_list_allquery.append(key)
          break
  
  #printing songs keys of matched songs
  print(img_list_allquery)
  images =img_list_allquery[:10]
  return images
