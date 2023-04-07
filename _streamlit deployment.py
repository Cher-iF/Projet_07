import nltk
nltk.download('words')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


import pandas as pd
import numpy as np
import regex as re
import string
import sklearn
import nltk.corpus

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

################  Text Cleaning   ######################################################
# Define stopwordss set
words = set(nltk.corpus.words.words())
stop = stopwords.words('english')
len(stop)
stop.remove("not")
len(stop)
# Define Lemmatizer   
lemmatizer = nltk.WordNetLemmatizer()
def remove_symbols(text):
    pattern = r'[' + string.punctuation + ']'
    return re.sub(pattern, '', text)
def lower_case(text):
    return text.lower()
def remove_extra_spaces(text):
    text = re.sub(' +', ' ', text)
    if text[-1] == ' ':
        text = text[:-1]
    if text[0] == ' ':
        text = text[1:]
    return text
def remove_numbers(text):
    return re.sub('[0-9]', '',text)
def remove_links(text):
    return re.sub(r'http\S+', '', text)
def remove_non_ASCII(text):
    return re.sub(r'[^\x00-\x7f]',r' ',text)
def tokenization(text):
    tokens = re.split('\W+',text)
    return tokens
def remove_stopwords(text_tokenized):
    txt_clean = [word for word in text_tokenized if word not in (stop)]
    return txt_clean
def lemmatization(text_cleaned):
    text_lemmatized = [lemmatizer.lemmatize(word) for word in text_cleaned if not word in set(stop)]
    return text_lemmatized

def clean_text(text):
    text = re.sub('\\n', '', text)
    text = remove_numbers(text)
    text = remove_symbols(text)
    text = lower_case(text)
    text = remove_non_ASCII(text)
    #text = remove_extra_spaces(text)
    text = remove_links(text)
    text = tokenization(text)
    text = remove_stopwords(text)
    text = lemmatization(text)
    text = ' '.join(text)
    return text
############################################################################################
from sklearn.feature_extraction.text import CountVectorizer
#import pickle
import joblib

#---------------       Models Loading -----------------------------------------------------#
cvFile='BoW.pkl'
#cv = pickle.load(open(cvFile,'rb'))
cv = joblib.load('BoW.pkl')
classifier = joblib.load('lr_classifier.pkl')

#---------------       Models on streamlit -----------------------------------------------------#
import streamlit as st
from PIL import Image

image = Image.open('classroom_logo.PNG')
image2 = Image.open('Air paradis.png')

st.sidebar.image(image, use_column_width=True)
st.image(image2, width=250)
st.sidebar.header("""
                  P7 Project on Sentiment Analysis
                  """)         
         
###################################### Main Input ###################################
tweet = st.text_area('Enter your Tweet')
#tweet = ' because , i am here to be here and not anywhere else &@@@@@'
df = pd.DataFrame({"tweet":[]})
#tweet_cleaned = clean_text(tweet)
df=df.assign(tweet=[clean_text(tweet)])
tweet_encoded = cv.transform(df['tweet'].values)
pred = classifier.predict(tweet_encoded)[0]
#pred

if st.button('Analyze'):
    with st.spinner('Analyzing the text …'):
        
        if pred == 0:
            st.success('Negative review')
        
        else :
            st.success('Positive review')
            st.balloons()
     



