#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# import libraries
import re
import pickle
import pandas as pd 
from sqlalchemy import create_engine
import nltk 
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,  GridSearchCV 
from sklearn.metrics import classification_report


nltk.download(['wordnet', 'punkt', 'stopwords'])


# load data from database
engine = create_engine('sqlite:///DisasterResponse.db')
engine.table_names()


df = pd.read_sql_table('DisasterResponse_table', con=engine)
X = df['message']
y = df.iloc[:,4:]

# ### 2. Write a tokenization function to process your text data

def tokenize(text):
    """
    Function: Split text into words
    Args:
      text(str): Your message 
    Return:
      lemmanization: a list of the root form of the message words
    """
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize text
    words = word_tokenize(text)
    
    # Remove stop words
    stop = stopwords.words("english")
    words = [t for t in words if t not in stop]
    
    # Lemmatization
    lemmanitazion = [WordNetLemmatizer().lemmatize(w) for w in words]
    return lemmanitazion


for message in X[:5]:
    tokens = tokenize(message)
    print(message)
    print(tokens, '\n')


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.


pipeline = Pipeline([
     ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(DecisionTreeClassifier()))
])


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train.shape, X_test.shape, y_train.shape, y_test.shape # Check that everything fits

pipeline.fit(X_train, y_train)


# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

y_pred = pipeline.predict(X_test)

for i, col in enumerate(y.columns):
    print(f'-----------------------{i, col}----------------------------------')
 
    print(classification_report(list(y_test.values[:, i]), list(y_pred[:, i])))


# ### 6. Improve your model
# Use grid search to find better parameters. 

# Parameters for the pipline
pipeline.get_params()

# Here we make a very simple grid search, this task take a while and patience (and a better computer)
parameters = {'clf__estimator__max_depth': [5, 10],
              'clf__estimator__min_samples_leaf':[2]}

cv = GridSearchCV(estimator=pipeline, param_grid=parameters)
cv.fit(X_train.as_matrix(), y_train)

# Show the best paramaters
print(cv.best_params_)

# Build a new model based on the best parameters
best_model = cv.best_estimator_
print (cv.best_estimator_)


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

y_pred = best_model.predict(X_test)

for i, col in enumerate(y.columns):
    print(f'-----------------------{i, col}----------------------------------')
 
    print(classification_report(list(y_test.values[:, i]), list(y_pred[:, i])))


# ### 9. Export your model as a pickle file

pickle.dump(cv, open('classifier.pkl', 'wb'))


# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

