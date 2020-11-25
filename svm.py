import pandas as pd
import numpy as np
import nltk
import csv
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('stopwords')
from collections import defaultdict
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report


def processing():
    #codigo tomado de los ejemplos de clase
    #leer el dataset
    Corpus = pd.read_csv("Data/data.csv",encoding='latin-1')
    
    #convertir mayusculas en minusculas
    for item in Corpus["0"]:
        Corpus["0"] = Corpus["0"].replace(item,item.lower())
    
    #tokenizacion
    Corpus['0'] = [word_tokenize(item) for item in Corpus["0"]]
    
    tag_map = defaultdict(lambda:wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    
    for index,item in enumerate(Corpus['0']):
        Final_words = []
        word_Lemmatized = WordNetLemmatizer()
        for word,tag in pos_tag(item):
            if word not in stopwords.words('english') and word.isalpha():
                word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                Final_words.append(word_Final)
        Corpus.loc[index,'text_final']=str(Final_words)
    
    #llamado a la funcion svm 
    supportMV(Corpus)
    


def supportMV(Corpus):
    #codigo tomado de los ejemplos de clase
    #separacion de entrenamiento y prueba
    X_train, X_test, y_train,y_test = train_test_split(Corpus["text_final"],Corpus['1'],test_size=0.3)
    
    #Transformar valores
    lab_enc = LabelEncoder()
    y_train_encoded = lab_enc.fit_transform(y_train)
    y_test_encoded = lab_enc.fit_transform(y_test)
    
    #TfidfVectorizer
    tfidf_vector = TfidfVectorizer(max_features=5000)
    tfidf_vector.fit(Corpus['text_final'])
    X_train_tfidf = tfidf_vector.transform(X_train)
    X_test_tfidf = tfidf_vector.transform(X_test)
    
    #modelo svm
    svmc=svm.SVC(C=1.0, kernel='linear')
    svmc.fit(X_train_tfidf, y_train_encoded)
    pred = svmc.predict(X_test_tfidf)
    pred = lab_enc.fit_transform(pred)
    print(pred)
    
    #mostrar metricas
    print(classification_report(pred,y_test_encoded))


def run():
    processing()

if __name__=="__main__":
    run()

