
import sys
import json
import ast
import time
import os
#input = sys.argv[1]



import subprocess



subprocess.check_call([sys.executable, "-m", "pip", "install", 'joblib'])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "scikit-learn"])
subprocess.check_call([sys.executable, "-m", "pip", "install", 'pandas'])
subprocess.check_call([sys.executable, "-m", "pip", "install", 'Flask'])
subprocess.check_call([sys.executable, "-m", "pip", "install", 'selenium'])
subprocess.check_call([sys.executable, "-m", "pip", "install", 'werkzeug'])

try:
  # Download Window requirement Lib
  subprocess.check_call([sys.executable, "-m", "pip", "install", 'pywin32'])
except:
  # Download Mac requirement Lib
  subprocess.check_call([sys.executable, "-m", "pip", "install", 'macos-notifications'])

subprocess.check_call([sys.executable, "-m", "pip", "install", 'plyer'])
subprocess.check_call([sys.executable, "-m", "pip", "install", 'nltk'])
subprocess.check_call([sys.executable, "-m", "pip", "install", 'regex'])

from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import *
import numpy as np
import webbrowser


try:
  # Import Window requirement Lib
  import win32api
  import win32com.client
except:
  # Import Mac requirement Lib
  from functools import partial
  from mac_notifications import client

import pythoncom
from plyer import notification





import nltk
nltk.download('stopwords') 
nltk.download('punkt') 
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('porter_test')
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words("english"))

import re

Folder = os.getcwd()

X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")
y_train = y_train['sentiment']
X_train = X_train.text
loaded_model = loaded_model = joblib.load("data/LinearSVC.joblib")

print("Please wait! Don't close the programe[0/1]")
vect = TfidfVectorizer(ngram_range=(1, 3)).fit(X_train)
print("Please wait! Don't close the programe[1/1]")
loaded_model.fit(vect.transform(X_train), y_train)
app = Flask(__name__, template_folder='')

def check_pos_neg(sentiment):
  if (sentiment == [1]):
    return "This is a negative review."
  else:
    return "This is a positive review."


def cleaning(text):
    text = text.lower() # change to lowercase
    text = re.sub('<br />', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r"https\S+|www\S+|http\S+", '', text, flags = re.MULTILINE)
    text = re.sub(r'\@w+|\#', '', text)
    text_tokens = word_tokenize(text)
    filtered_tokens = []
    for w in text_tokens:
      if not w in stop_words or w == 'not':
        filtered_tokens.append(w)
    return " ".join(filtered_tokens)

def stem(text):
  stemmer = PorterStemmer()
  tokens = word_tokenize(text)
  words = [stemmer.stem(word) for word in tokens]
  data = " ".join(words)
  return data

@app.route('/')

def index():
    return render_template('./index.html')


@app.route('/ans', methods = ['POST'])
def ans():
    if request.method == 'POST':
        print(request.form)
        ls = list(request.form.lists())
        value = ls[0][1]
        review = value[0]

        print(value)
        print(review)

        review = stem(cleaning(review))
        

        print(review)
        value = [review]

        result = loaded_model.predict(vect.transform(value))
        message = check_pos_neg(result)

        try:
          # Notify Win
          notification.notify(title = "RESULT", message = message, timeout = 10)

        except:
          # Notify Mac
          client.create_notification(
            title="Result",
            subtitle= message,
            action_button_str="OK"
          )



        try:
          # Win message box
          result = win32api.MessageBox(None, message, "Result", 1)
          if result == 1:
              print('Ok')
          elif result == 2:
              print('cancel')
        except:
          # Mac message box
          os.system("""osascript -e 'Tell application "System Events" to display dialog """+ message +""" with title "Result"'""")



          
          
        print("result", message)
        return ('', 204)
    


if __name__ == '__main__':
    print("Please go to the website 127.0.0.1:5001")
    try:
       subprocess.check_call('explorer "http://127.0.0.1:5001"', shell=True)
    except:
       print("Done try Win")
    try:
       subprocess.check_call('open http://127.0.0.1:5001', shell=True)
    except:
       print("Done try Mac")
    app.run(
        host = '127.0.0.1',
        port = 5001,
        debug= False
    )


