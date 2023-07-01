# Disaster Response Pipeline

This project aims to build a web application that can automatically classify disaster-related messages. The project utilizes a dataset containing messages in different categories such as water, food, shelter, medicine, and more. These messages are used to train a machine learning model that can classify new messages into their respective categories.

## Project Components

The Disaster Response Pipeline project consists of the following components:

1. **ETL Pipeline**: This component is responsible for data preprocessing. It involves cleaning the data, performing necessary transformations, and storing the processed data in a database.

2. **ML Pipeline**: The ML Pipeline component involves training a machine learning model on the preprocessed data. The model is trained using techniques from natural language processing (NLP) and classification. After training, the model is saved for later use.

3. **Web Application**: The web application component provides a user interface for interacting with the trained model. Users can enter messages, which are then classified into relevant categories by the model. The application also displays visualizations of the data and categories.# Disaster-Response-Pipeline
Udacity project

## Libraries 
All libraries are available in Anaconda distribution of Python. The used libraries are:

- pandas
- re
- sys
- json
- sklearn
- nltk
- sqlalchemy
- pickle
- Flask
- plotly
- sqlite3
- The code should run using Python versions 3.*.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Go to http://127.0.0.1:5000/ (Host and port used by Flask for default)

## Acknowledgments

This project is part of the Udacity Data Scientist Nanodegree program. The dataset used in this project is provided by Udacity, and the project template, including the web application, is provided by Udacity as well.
