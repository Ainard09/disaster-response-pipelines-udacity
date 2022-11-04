# Disaster Response Pipeline Project
## Table of Content
---
1. [Instructions](#instructions)
2. [Summary](#summary)
3. [File Description](#file-description)
4. [Dataset](#dataset)
5. [Modeling Process](#modeling-process)
6. [Images](#Images)
7. [Effect of Imbalance](#effect-of-imbalance-data)
8. [Model Results](/notebooks/ML%20Pipeline%20Preparation.ipynb)

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

* To run ETL pipeline that cleans data and stores in the database `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
* To run ML pipeline that trains classifier and saves `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
2. Run the following command in the app's directory to run your web app. `python run.py`

### Summary:
In this project, I've analyzed disaster data from Figure Eight to build a model for an API that classifies disaster messages.

The data set contains real messages that were sent during disaster events. I've created a machine learning pipeline to categorize these events so that messages can be sent to an appropriate disaster relief agency.

This project has an app inside the app folder. Using it an emergency worker can input a new message and get classification results in several categories. The web app also display the visualization of the data.

### File Description:
ETL Pipeline Preparation.ipynb: Notebook contains ETL Pipeline.
ML Pipeline Preparation.ipynb: Notebook contains ML Pipeline.
categories.csv: Categories data set.
messages.csv: Messages data set.
classifier.pkl: Trained model pickle file.
train_classifier.py: Python file for model training.
metrics_evaluator.py: python file for model metric evaluation
disaster_categories.csv: Disaster Categories data set.
disaster_messages.csv: Disaster Messages data set.
process_data.py: Python ETL script.
app: Flask Web App
run.py: Flask Web App main script.
images: Image Folder
Profile: file used by heroku for deployment of app
requirements.txt: Text file containing list of packages used.
LICENSE: Project LICENSE file.

### Dataset

This disaster data is from [Figure Eight.](https://appen.com/) This dataset has two files messages.csv and categories.csv.

### Data Cleaning
Based on id two datasets were first merged into df. Categories were split into separate category columns. Category values were converted to numbers 0 or 1. Replaced categories column in df with new category columns. Removed duplicates based on the message column.
df were exported to DisasterResponse.db.

### Modeling Process
Wrote a tokenization function to process text data.
Build a machine learning pipeline using CountVectorizer, TfidfTransformer, RandomForestClassifier, and Pipeline.
Split the data into training and test sets.
Using pipeline trained and evaluated a simple RandomForestClassifier.
Then using hyperparameter tuning with 5 fold cross-validation fitted 100 models to find the best random forest model for predicting disaster response category. Random Forest best parameters were `{'clf__min_samples_split': [3],'clf__n_estimators': [5]}`
Using this best model we've made train_classifier.py

### Images
![image result](/images/categories_charts.png)
![image result](/images/model_predicts.png)

### Effect of Imbalance data

![image result](/images/catagories_counts.png)

