
#
# @Function:This function was used to extract twitter text from 
#           twitter.csv file 
#
# @Return: This function will return an array in unicode form
#          

import csv

DATA_DIR = './training_data/twitter.csv'

def twitter_text_extractor():
    
    with open(DATA_DIR,'rb') as data_file:
        reader    = csv.reader(data_file,delimiter='\t')
        data_text = []
        for row in reader:
            try:
                tweet = row[5]
            except:
                tweet = ""
            data_text.append(unicode(tweet,"utf-8")) 
        
    return data_text

