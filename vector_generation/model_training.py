
# 
# @Function: This script was set to generate a model for 
#            vector generation purpose using Google word2vec
#
# @Return: The model will be saved at certain space as training
#
# @Concerns: This function is related to news and HTML stuff 
#            Really wonders if it will work fine under twitter data
#

from sklearn.datasets import fetch_20newsgroups
from gensim.models import word2vec
from bs4 import BeautifulSoup
import re
import nltk
import time

# Config
MODEL_STORAGE = '../model_storage'
feature_vector_dimension = 300
minimum_word_considerate = 20
concurrent_cpu_amount    = 2
context_window           = 5
down_sampling_parameter  = 1e-3


#
# @Function: This function is very useful since it will convert from a HTML link
#            First grab useful information from this link and discard string like '</head>'
#            Then tokenize the remaining string. From multiple sentences to a sentence array
#
# @Parameter: Raw news data, which may contain HTML stuff and consists of multiple sentences
#
# @Return:  Sentence array, contain many sentence in this array.
#

def news_to_sentences(news):
    news_text = BeautifulSoup(news).get_text()
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizers.tokenize(news_text)
    sentences = []
    for sent in raw_sentences:
        sentences.append(re.sub('[^a-zA-Z]', ' ', sent.lower().strip()).split())
    return sentences


if(__name__=='__main__'):
    
    news = fetch_20newsgroups(subset = all)
    sentence_array = []
    
    for one_news in news.data:
        sentence.extend(news_to_sentences(one_news))

    model = word2vec.Word2vec(sentences,
                              workers   = concurrent_cpu_amount,
                              size      = feature_vector_dimension,
                              min_count = minimum_word_considerate,
                              window    = context_window,
                              sample    = down_sampling_parameter)    
    
    model.init_sims(replace=True)
    model.save(MODEL_STORAGE)

