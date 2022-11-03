import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """ Function to load csv file from filepath.
    Args:
        message_filepath(string) csv file containing message data
        categories_filepath(string) csv file conatining the categories data
   
    Return:
        df(dataframe) dataframe of merged message and categories data
    """
    message= messages = pd.read_csv(messages_filepath)
    categories= categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on=('id'))
    
    return df

def clean_data(df):
    """ Function to clean the merged dataframe.
    Args:
        df(dataframe) merged message and categories data
    
    Return:
        df(dataframe) cleaned dataframe
    """
    categories = df['categories'].str.split(';',expand=True)
    
    # select the first row of the categories dataframe
    rows = categories.loc[0]

    # use this row to extract a list of new column names for categories.
    category_colnames = rows.apply(lambda x: x[:-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
   
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
    
        # convert column from string to numeric
        categories[column] = categories[column].str.replace('2','1')
        categories[column] = pd.to_numeric(categories[column])
    
    # drop the original categories column from `df`
    df= df.drop('categories',axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1)
    
    # drop duplicates
    df= df.drop_duplicates()
    
    return df 

def save_data(df, database_filename):
    """Function to save cleaned df to file.
    Args:
        df(dataframe) cleaned dataframe
        database_filename(database) filepath of database
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('disaster_response1', engine, index=False) 

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()