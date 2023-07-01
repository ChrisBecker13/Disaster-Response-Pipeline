import sys
import pandas as pd
import os
import re
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import nltk
nltk.download(['wordnet', 'punkt', 'stopwords'])

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sqlalchemy import create_engine

def load_data(database_filepath):
    """
    INPUT:
    database_filepath - 
    
    OUTPUT:
    X - messages (input variable) 
    y - categories of the messages (output variable)
    category_names - category name for y
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterResponse', engine)    
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = Y.columns
    return X, Y, category_names
 

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

def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(DecisionTreeClassifier()))
    ])
    parameters = {'clf__estimator__max_depth': [5, 7],
                  'clf__estimator__min_samples_leaf':[2]}
    cv = GridSearchCV(estimator=pipeline, param_grid=parameters)
    return cv
# Evaluate model
def evaluate_model(model, X_test, Y_test, category_names):
    """
    INPUT:
    model - ML model
    X_test - test messages
    y_test - categories for test messages
    category_names - category name for y
    
    OUTPUT:
    none - print scores (precision, recall, f1-score) for each output category of the dataset.
    """
    Y_pred_test = model.predict(X_test)
    for i, col in enumerate(category_names):
        print(f'-----------------------{i, col}----------------------------------')
        print(classification_report(list(Y_test.values[:, i]), list(Y_pred_test[:, i])))  
    
def save_model(model, model_filepath):
        """
        INPUT:
        model - ML model
        model_filepath - location to save the model
    
        OUTPUT:
        none
        """
        with open(model_filepath, 'wb') as f:
            pickle.dump(model, f)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        print('Building model...')
        model = build_model()
        print('Training model...')
        model.fit(X_train, Y_train)
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)
        print('Trained model saved!')
    else:
        print('Please provide the filepath of the disaster messages database '\
        'as the first argument and the filepath of the pickle file to '\
        'save the model to as the second argument. \n\nExample: python '\
        'train_classifier.py ../data/DisasterResponse.db classifier.pkl')
if __name__ == '__main__':
    main()