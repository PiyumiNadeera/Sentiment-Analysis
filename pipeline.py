import pandas as pd
import numpy as np
import re
import string
import pickle

with open('static/model/model.pickle','rb') as file:
    model = pickle.load(file)

with open('static/model/corpora/stopwords/english','r') as file:
    stop_words = file.read().splitlines()

vocab = pd.read_csv('static/model/vocabulary.txt',header=None)
tokens = vocab[0].tolist()

from nltk.stem import PorterStemmer
ps = PorterStemmer()

def remove_punctuation(txt):
    for punctuation in string.punctuation:
        txt = txt.replace(punctuation,'')
    return txt

def preprocessing(txt):
    data = pd.DataFrame([txt],columns=['tweet'])
    data['tweet'] = data['tweet'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    data['tweet'] = data['tweet'].apply(lambda x: " ".join(re.sub(r'^gttps?:\/\/.*[\r\n]*','',x,flags=re.MULTILINE) for x in x.split()))
    data['tweet'] = data['tweet'].apply(remove_punctuation)
    data['tweet'] = data['tweet'].str.replace(r'\d+','',regex=True)
    data['tweet'] = data['tweet'].apply(lambda x:" ".join(x for x in x.split() if x not in stop_words))
    data['tweet'] = data['tweet'].apply(lambda x:" ".join(ps.stem(x) for x in x.split()))
    return data['tweet']

def vectorizer(dataset):
    vectorized_list = []
    
    for sentence in dataset:
        sentence_list = np.zeros(len(tokens))
        for i in range(len(tokens)):
            if tokens[i] in sentence.split():
                sentence_list[i] = 1
        vectorized_list.append(sentence_list)
    vectorized_list_new = np.array(vectorized_list,dtype=np.float32)
    return vectorized_list_new

def get_prediction(vectorized_text):
    prediction = model.predict(vectorized_text)
    if prediction == 1:
        return 'Positive'
    else:
        return 'Negative'
    