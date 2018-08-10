
# 
# @Function: This script was set to generate a model for 
#            vector generation purpose using Google word2vec
#
# @Return: The model will be saved at certain space as training
#

from sklearn.datasets import fetch_20newsgroups
from gensim.models import word2vec
from bs4 import BeautifulSoup
import re
import nltk
import time

MODEL_STORAGE = '../model_storage'

