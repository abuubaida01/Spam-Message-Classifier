import sklearn as sk
import streamlit as st
import pickle  # for reading and dumping models
import nltk #for many function used below
from nltk.corpus import stopwords ## for stops words like is the of etc
from nltk.stem.porter import PorterStemmer ## for converting to Dic form
ps = PorterStemmer() 
import pandas as pd
import string ## for getting access to English most common punctuations

## One Way of loading Pickle file 
## let's import our modules
# tfidf = pickle.load(open('Vectorizer.pkl', 'b'))
# model = pickle.load(open('model.pkl', 'b'))
# preprocessing = pickle.load(open("preprocessing"))
## by this method we can read our pickle modules

### Transformer function for preprocessing 
def Transform_text(text):
   T = text.lower()  
   T = nltk.word_tokenize(text) ## for getting single single words

   ## for removing special characters 
   List=[]   # to store all char
   for i in T:
      if i.isalnum(): ## just alphabet and numbers are allowed
         List.append(i)
   T = List[:] ## as list is immutable so will have to copy like this or .copy()
   List.clear() # to use it again
   
   ## let's check for other conditions 
   for i in T:
      if i not in stopwords.words('english') and i not in string.punctuation:
         List.append(i)
   text = List[:]; List.clear()

   ## for stemming 
   for i in text:
      List.append(ps.stem(i))
   return " ".join(List)  ## to get as a string of list


vect = pd.read_pickle("Vectorizer.pkl")
model = pd.read_pickle("model.pkl")
# preprocess = pd.read_pickle("preprocessing.pkl")

## body of web page
st.title("\n Email Spam Detector")

# getting input 
data = st.text_area("Paste/Write your Email")

# let's add button 
if st.button("PREDICT"):
   # Step one 
   # Trans = preprocess([data])
   transformed = Transform_text(data)
   # st.header(transformed)

   # step Two Vectorization
   Vector = vect.transform([transformed])
   # st.header(Vector)
   #Step three Prediction
   Result = model.predict(Vector)[0] #There are two result 0 or 1, we move 0
   # st.header(Result)
   #condition 
   
   if Result ==1 :
      st.header("It is a Spam E-Mail :( ")
   else:
      st.header("It's not a Spam E-Mail :)")


