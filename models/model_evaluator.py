import pandas as pd
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score

def get_metrics(test_value, predicted_value):
    """
    get_metrics calculates f1 score, accuracy and recall
    Args:
        test_value (list): list of actual values
        predicted_value (list): list of predicted values
    Returns:
        dictionray: a dictionary with accuracy, f1 score, precision and recall
    """
    accuracy = accuracy_score(test_value, predicted_value)
    precision = precision_score(
        test_value, predicted_value, average='micro')
    recall = recall_score(test_value, predicted_value, average='micro')
    f1 = f1_score(test_value, predicted_value, average='micro')
    
    return {'Accuracy': accuracy, 'f1 score': f1, 'Precision': precision, 'Recall': recall}



def compute_evaluation(Y_test,Y_train,predict_train,predict_test):
    """" function to evaluate each category dataset
    Args:
        Y_test(dataframe): dataframe for test dataset
        Y_train(dataframe): dataframe for train dataset
        predict_train(list): list of train predicted values
        predict_test(list): list of test predicted values
    Returns:
        metrics_train_df(df): train dataframe of evaluation metrics
        metrics_test_df(df): test dataframe of evaluation metrics
    """
    # get the accuracy score for each category on train dataset
    metric_train= []
    for i, column in enumerate(Y_test.columns):
        metrics= get_metrics(Y_train.loc[:,column].values, predict_train[:,i])
        metric_train.append(metrics)
    
    result_train_df= pd.DataFrame(metric_train)
    
    
    # get the accuracy score for each category on test dataset
    metric_test= []
    for i, column in enumerate(Y_test.columns):
        metrics = get_metrics(Y_test.loc[:, column].values, predict_test[:, i])
        metric_test.append(metrics)
    
    result_test_df= pd.DataFrame(metric_test)
    
    return result_train_df, result_test_df