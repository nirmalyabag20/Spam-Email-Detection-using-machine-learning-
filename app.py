
import streamlit as st
import pickle

import string

import nltk
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


tfidf=pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))


st.title("Mail Spam Classifier")

def transform_text(text):
    text = text.lower()  # Lowercase()
    text = nltk.word_tokenize(text)  # Tokenize

    l = []
    for i in text:  # Remove special characters
        if i.isalnum():
            l.append(i)

    text = l[:]
    l.clear()
    for i in text:  # Remove stopwords and punctuation
        if i not in stopwords.words('english') and i not in string.punctuation:
            l.append(i)

    text = l[:]
    l.clear()
    for i in text:  # Stemming
        l.append(ps.stem(i))


    return " ".join(l)

input_sms= st.text_area("Enter the message")

if st.button('PREDICT'):

    transformed_sms=transform_text(input_sms)

    vector_input = tfidf.transform([transformed_sms])

    result= model.predict(vector_input)[0]

    if result == 1 :
        st.header("SPAM")
    else:
        st.header("NOT SPAM")









