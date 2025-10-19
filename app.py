import streamlit as st
import pickle 
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
# from sklearn.naive_bayes import MultinomialNB
import sklearn

ps=PorterStemmer()

def text_transform(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)

    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text=y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)

vectorizer=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('spam-model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms=st.text_input("Enter the message")

if st.button("Predict"):
    #1.preprocessing
    transformed_sms=text_transform(input_sms)

    # 2.vectorize
    vector_input=vectorizer.transform([transformed_sms])

    # 3.predict
    result=model.predict(vector_input)[0]

    # 4.display
    if result==1:
        st.header("Spam")
    else:
        st.header("Not Spam")