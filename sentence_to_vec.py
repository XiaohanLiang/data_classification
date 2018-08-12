
#
# @Function:This script is designed to trnasform sentences to a 
#           100 dimensional vector
#
# @Input: Input a filename which indicates the place for your 
#         text files that you would like to deploy transformation
#
# @Output: The text will be transformed and stored in to some 
#          certain place
#

from gensim.models import Word2Vec
import numpy as np
import jieba
import re
model = Word2Vec.load("./model_storage/model")
saveDir = ""

def split_text_file(file_name):
    raw_data = []
    w = open(filename,'r',encoding='utf-8')
    for line in w.readlines():
        newline = line.strip()
        newline = re.sub(' ','',newline)
        newline = jieba.cut(newline)
        raw_data.append(list(newline))
    w.close()
    return raw_data

#
# @Function: From this function when we input a sentence 
#            in the word list form we would get a vector 
#            back which is the vector of this sentence
#
#

def sentence_to_vec(sentence):
    sentence_vector = []
    for word in sentence:
        try:
            sentence_vector.append(model.wv[word])
        except:
            continue
    sentence_vector = np.array(sentence_vector)
    v = sentence_vector.sum(axis = 0)
    return v/np.sqrt((v ** 2).sum())


