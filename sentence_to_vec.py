
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
from bs4 import BeautifulSoup
import numpy as np
import re
import sys

sys.dont_write_bytecode = True
params = eval(open("settings.txt").read())
model = Word2Vec.load(params['model_path'])
file_name = params['twitter_path']

def tweets_to_sentences(data):
    data_text = BeautifulSoup(data).get_text()
    sentences = re.sub('[^a-zA-Z]', ' ', data_text.lower().strip()).split()
    return sentences

def split_text_file(file_name):
    raw_data = []
    w = open(file_name,'r')
    for line in w.readlines():
        newline = tweets_to_sentences(line)
        raw_data.append(newline)
    w.close()
    return raw_data


#
# @Function: From this function when we input a sentence 
#            in the word list form we would get a vector 
#            back which is the vector of this sentence
#
#

def sentence_to_vec():
    sentence_array = split_text_file(file_name)
    return_ans = []
    for sentence in sentence_array:
        sentence_vector = []
        for word in sentence:
            try:
                sentence_vector.append(model.wv[word])
            except:
                continue
        sentence_vector = np.array(sentence_vector)
        v = sentence_vector.sum(axis = 0)
        ans =  v/np.sqrt((v ** 2).sum())
        return_ans.append(ans)
    return return_ans

