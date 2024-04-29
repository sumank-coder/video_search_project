import Img_TFIDF
import nltk
import math
#nltk.download('all')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

#pre-processing for query
def pre_process_query(query):
    processed_query = []
    stop_words = set(stopwords.words('english'))
    for data in query.split():
        low_val = data.lower()
        lemm_query = WordNetLemmatizer().lemmatize(low_val)
        processed_query.append(lemm_query)
    l = []
    for data in processed_query:
      if data.isalpha():
        if data not in stop_words and len(data)>1:
          l.append(data)
    less_wordy_query = l
    return less_wordy_query

def calc_Denom1_img(tfIdf):
    val=0
    denominator_doc = 0
    for term,value in tfIdf.items():
        val += math.pow(value, 2)
        denominator_doc = math.sqrt(val)
    return denominator_doc

# cosine similarity b/w query and image
def queryImg_cosine(img_TFIDF,query):
    denom2_query = math.sqrt(len(query))
    query_img_score={}
    for doc, tfIdf in img_TFIDF.items():
        denom1_doc = calc_Denom1_img(tfIdf)
        score = 0
        score_final = 0
        for term in query:
            if term in tfIdf:
                score += (tfIdf[term])
        score_final= (score)/ (denom2_query * denom1_doc)
        query_img_score[doc] = score_final
    sorted_query_img_score = dict(sorted(query_img_score.items(), key=lambda item: item[1], reverse = True))
    return sorted_query_img_score

def Top10_image_list(results):
    count =1
    img_list = []
    for image,score in results.items():
        if score != 0:
            img_list.append(image)
            print(count,"\t",score, "\t", image)
            count+=1
            if(count >=4):
                break
    return img_list


def Image_search(query):
    query_processed = pre_process_query(query)
    img_query_cosine_simil = queryImg_cosine(Img_TFIDF.img_desc_TFIDF, query_processed)
    images = Top10_image_list(img_query_cosine_simil)
    print(images)
    #d_video = ['images/abc.mp4']
    return images
    #return d_video

'''

if __name__ == "__main__":
    query = "water body"
    query_processed = pre_process_query(query)
    img_query_cosine_simil = queryImg_cosine(Img_TFIDF.img_desc_TFIDF, query_processed)
    print_top10_results(img_query_cosine_simil)

'''