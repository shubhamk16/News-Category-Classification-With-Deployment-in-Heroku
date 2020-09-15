import nltk
import pandas as pd
#nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
#import numpy as np
import re
import pickle
from nltk.corpus import stopwords
nltk.download('wordnet')
#%%
# import os 
# datadir = "D:\OneDrive - Subex Limited\Competitions\MachineHack News Category Classification Hackathon\Machinehack_News_Category"
# os.chdir(datadir)
#%%
data = pd.read_csv("Data_train.csv")

X = data['STORY']
y = data['SECTION']

#%%
from nltk.stem import WordNetLemmatizer
stemmer = WordNetLemmatizer()
documents = []

for i in range(0, len(X)):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(X.iloc[i]))
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    # Converting to Lowercase
    document = document.lower()
    # Lemmatization
    document = document.split()
    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    
    documents.append(document)
#%%
# df= pd.read_csv("spam.csv", encoding="latin-1")
# df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
#  # Features and Labels
# df['label'] = df['class'].map({'ham': 0, 'spam': 1})
# X = df['message']
# y = df['label']
 	
#  	# Extract Feature With CountVectorizer
# cv = CountVectorizer()
# X = cv.fit_transform(X) # Fit the Data

# pickle.dump(cv, open('tranform.pkl', 'wb'))


# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# #Naive Bayes Classifier
# from sklearn.naive_bayes import MultinomialNB

# clf = MultinomialNB()
# clf.fit(X_train,y_train)
# clf.score(X_test,y_test)
# filename = 'nlp_model.pkl'
# pickle.dump(clf, open(filename, 'wb'))

#%%
data['Documents'] = documents
from sklearn.feature_extraction.text import CountVectorizer

# Extract Feature With CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(data['Documents']) # Fit the Data

pickle.dump(cv, open('News_Vectorizer.pkl', 'wb'))

#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)

filename = 'Nlp_model.pkl'
pickle.dump(clf, open(filename, 'wb'))







