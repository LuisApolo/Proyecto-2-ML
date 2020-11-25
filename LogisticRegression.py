import pandas as pd
from collections import defaultdict
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

def processing():
    Corpus = pd.read_csv("Data/data.csv", encoding='latin-1')
    
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
    logisticRegression(Corpus) 


def logisticRegression(Corpus):
    #
    X_train, X_test, y_train, y_test = train_test_split(Corpus['text_final'], Corpus['1'],test_size = 0.3)
    
    
    lab_enc = LabelEncoder()
    y_train_encoded = lab_enc.fit_transform(y_train)
    y_test_encoded = lab_enc.fit_transform(y_test)
    
    #TfidfVectorizer
    vectorizer = TfidfVectorizer()
    vectorizer.fit(Corpus['text_final'])
    X_train_tfidf = vectorizer.transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    #modelo regresion logistica
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train_encoded)
    y_pred = model.predict(X_test_tfidf)
    y_pred = lab_enc.fit_transform(y_pred)
    print(y_pred)
    
    #mostrar métricas
    print(classification_report(y_pred, y_test_encoded))
    

def run():
    processing()


if __name__ == "__main__":
    run()
