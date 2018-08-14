## How the model has been generated

### Ways proposed by official website:
- demo.sh will invoke create-text8-vecotr.sh 
- check is there is existing model [text8]
    - if there is,exit
- if model-text8 does not exits,download zip file form certain website
- extract zip file and now we have the basic data origins
- ways of training model, listed in "create-text8-vector-data.sh"
    - bin/word2vec/ -train data/text8 output ....

### Ways proposed in other people's code
- import dataset from sklearn and use it as original data
- Using tools like 'tokenizer' 'ntlk' 'Beautifulsoup' to grab maybe cleaning data
- Change the sentence into vocabulary list form
- By now we have the whole training data set ready
- Set parameter and begin training,save model afterwards
- Using some tool to again calculate the similarity of sentences and vocabularies

-----------------------------------------------------------------------------------

## It is always worth mentioning that..
So far we have already get the vector of each vocabulary or each sentence.
What's more important is that, we can almost sure that similar sentence
will be classified into the same group.

Assume the similar sentence will be classified into the same group then
the only thing to worry about is, if SpiNNaker will classify each of them
correctly.

If it also will, every one is happy now.
