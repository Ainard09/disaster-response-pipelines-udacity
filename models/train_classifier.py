import sys
import re
import pandas as pd
import nltk
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import pickle
from model_evaluator import get_metrics, compute_evaluation

nltk.download('wordnet')  # download for lemmatization
nltk.download('stopwords')
nltk.download('punkt')

def load_data(database_filepath):
    """ Function to load database file from filepath.
    INPUT:
        database_filepath(string) sql database containing the data
    """
    #read from the data using sqlalchemy
    engine= create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('disaster_response1',engine)
    X = df['message']
    Y= df.drop(['id','message','original','genre'], axis=1)
    category_names= Y.columns
    
    return X,Y,category_names


def tokenize(text):
    """ Function to clean the texture data.
    INPUT:
        text(string) texts that want to be cleaned
        
    OUTPUT:
        clean_tokens(list) list of words that has been cleaned
    """
    
    text= re.sub(r"[^a-zA-Z0-9]"," ",text)
     # tokenize text
    tokens = word_tokenize(text) 
    tokens= [tok for tok in tokens if tok not in stopwords.words('english')]
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case
        clean_tok = lemmatizer.lemmatize(tok,pos='v').lower()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """ Function to build a pipeline model.
    OUTPUT:
         model(string) tuned model with gridsearch 
     """
    #instantiate Pipeline class
    pipeline = Pipeline([
    ('vect',CountVectorizer(tokenizer=tokenize)),
    ('tfidf',TfidfTransformer()),
    ('clf',RandomForestClassifier())
    ])
    
    
    #define parameters for tuning the pipeline model
    parameters = {
    'clf__n_estimators': [100],
    'clf__min_samples_split': [3]
    }

    model = GridSearchCV(pipeline, param_grid=parameters,cv=2, verbose=3)
    
    return model


def evaluate_model(model, X_test, Y_test, X_train,Y_train, category_names):
    """ Function to evaluate how the model perform.
    INPUT:
        model(string) the tuned model with gridsearch
        X_test(dataframe) the testing df data used by model to make a prediction
        Y_test(datframe) the 36 categories dataset
        categories_names(list) list of categories columns
    """
    #make a prediction     
    y_preds_train= model.predict(X_train)
    y_preds_test= model.predict(X_test)
    
    trains_results_df,test_results_df= compute_evaluation(Y_test,Y_train,y_preds_train,y_preds_test)
    
    print(test_results_df)
    
    
def save_model(model, model_filepath):
    """ Function to save the model to a pickle file.
    INPUT:
        model(string) the tuned model with gridsearch
        model_filepath(string) filepath to save the model
    """
    model = model.best_estimator_
    with open(model_filepath, 'wb') as file:
        model= pickle.dump(model,file)


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
        evaluate_model(model, X_test, Y_test,X_train,Y_train,category_names)

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