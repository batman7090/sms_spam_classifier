import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    buffer = []
    for i in text:
        if i.isalnum():
            buffer.append(i)
            
    text = buffer[:]
    buffer.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            buffer.append(i)

    text = buffer[:]
    buffer.clear()

    for i in text:
        buffer.append(ps.stem(i))
    return " ".join(buffer)


tfidf = pickle.load(open("artifacts/vectorizer.pkl", "rb"))
model = pickle.load(open("artifacts/model.pkl", "rb"))


st.title("Email/SMS Spam Classifier")

input_message = st.text_area("Enter the message")

if st.button("Predict"):
    # 1. Preprocess
    transformed_text = transform_text(input_message)

    # 2. vectorize
    vectorized_input = tfidf.transform([transformed_text])

    # 3. predict
    prediction = model.predict(vectorized_input)

    # 4. Display
    if prediction == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")