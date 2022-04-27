import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import f_regression, RFE


############################ ACQUIRE ############################

def acquire_heart_data():
    '''
    This function will acquire the data by reading the .csv saved and return a dataframe
    '''
    df = pd.read_csv('heart_2020_cleaned.csv')
    print('Dataframe Shape')
    print(df.shape)
    print(' ')
    return df


############################ PREPARE ############################

def prep_data():
    '''
    This function will acquire the heart data and then prepare it by
    renaming columns and encoding columns
    '''
    # Acquire data using .csv
    df = pd.read_csv('heart_2020_cleaned.csv')
    # Rename columns for readability
    df = df.rename(columns = {'HeartDisease':'heart_disease', 'Smoking':'smoking', 'AlcoholDrinking':'alcohol_drinking', 'Stroke':'stroke', 'PhysicalHealth':'physical_health', 'MentalHealth':'mental_health',
                          'DiffWalking':'diff_walking','Sex':'sex', 'AgeCategory':'age_category', 'Race':'race', 'Diabetic':'diabetic', 'PhysicalActivity':'physical_activity','GenHealth':'gen_health',
                          'SleepTime':'sleep_time', 'Asthma':'asthma', 'KidneyDisease':'kidney_disease','SkinCancer':'skin_cancer'})
    # Get dummies for columns that have two values and dropping first
    dummy_df = pd.get_dummies(df[['heart_disease', 'smoking', 'alcohol_drinking', 'stroke', 'diff_walking','sex', 'physical_activity', 'asthma', 'kidney_disease', 'skin_cancer']], dummy_na=False, drop_first=[True, True])
    # another dummy_df but not dropping first so i can see all values
    dummy_df1 = pd.get_dummies(df[['race','diabetic']])
    # Encoding the age category
    cat_col = ['age_category']
    for col in cat_col:
        df[col] = df[col].map({'18-24':0, '25-29':1,'30-34':2,'35-39':3,'40-44':4,'45-49':5,'50-54':6,'55-59':7,'60-64':8,'65-69':9,'70-74':10,'75-79':11,'80 or older':12})
    cat_col2 = ['gen_health']
    # Encoding the gen health category
    for col in cat_col2:
        df[col] = df[col].map({'Poor':0,'Fair':1,'Good':2,'Very good':3,'Excellent':4})
    # Creating a new df that is made up of encoded columns
    new_df = pd.concat([df, dummy_df, dummy_df1], axis = 1)   
    # Dropping columns that are no longer useful
    new_df = new_df.drop(columns = ['heart_disease', 'smoking', 'alcohol_drinking', 'stroke', 'diff_walking', 'sex', 'race', 'diabetic', 'physical_activity', 
                            'asthma', 'kidney_disease', 'skin_cancer'])
    print('New Dataframe Shape')
    print(new_df.shape)
    print(' ')
    return new_df

############################ SPLIT DATA ############################

def split_data(new_df):
    '''
    This function takes in a dataframe and splits the data into train, validate and test samples. 
    Test, validate, and train are 20%, 24%, & 56% of the original dataset, respectively. 
    The function returns train, validate and test dataframes. 
    '''
    # split dataframe 80/20, stratify on churn to ensure equal proportions in both dataframes
    train_validate, test = train_test_split(new_df, test_size=.2, 
                                            random_state=123, 
                                            stratify=new_df.heart_disease_Yes)
    # split previous larger dataframe by 70/30, stratify on churn
    train, validate = train_test_split(train_validate, test_size=.3, 
                                       random_state=123, 
                                       stratify=train_validate.heart_disease_Yes)
    # results in 3 dataframes
    return train, validate, test

############################ DOWNSAMPLE ############################

# Data is downsampled to balance adults without heart disease and adults with heart disease
def downsample_data(new_df):
    class_0 = new_df[new_df['heart_disease_Yes'] == 0]
    class_1 = new_df[new_df['heart_disease_Yes'] == 1]
    class_0 = class_0.sample(len(class_1), replace = True)
    new_df = pd.concat([class_0, class_1], axis = 0)
    print(new_df.shape)
    print('Heart Disease Counts in Train dataframe')
    print(' ')
    print(new_df.heart_disease_Yes.value_counts())
    return new_df

    ############################ SCALE DATA ############################
# data must be scaled for modeling
def scale_data(train, validate, test, return_scaler=False):
    '''
    Scales the 3 data splits.
    
    takes in the train, validate, and test data splits and returns their scaled counterparts.
    
    If return_scaler is true, the scaler object will be returned as well.
    '''
    columns_to_scale = ['BMI', 'mental_health', 'sleep_time', 'physical_health']
    
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    
    scaler = MinMaxScaler()
    scaler.fit(train[columns_to_scale])
    
    train_scaled[columns_to_scale] = scaler.transform(train[columns_to_scale])
    validate_scaled[columns_to_scale] = scaler.transform(validate[columns_to_scale])
    test_scaled[columns_to_scale] = scaler.transform(test[columns_to_scale])
    
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    else:
        return train_scaled, validate_scaled, test_scaled

############################ X_TRAIN Y_TRAIN ############################

def get_Xtrain_ytrain(train_scaled, train, validate, validate_scaled, test, test_scaled ):
    # Create X and y version of train, y is a series of the target variable and X are all the features. 
    drop_cols = ['BMI', 'mental_health', 'heart_disease_Yes', 'sleep_time','alcohol_drinking_Yes', 'sex_Male', 'physical_activity_Yes', 'asthma_Yes', 
                 'skin_cancer_Yes', 'race_American Indian/Alaskan Native', 'race_Asian', 'race_Black', 'race_Hispanic', 'race_Other', 
                 'race_White', 'diabetic_No', 'diabetic_No, borderline diabetes', 'diabetic_Yes (during pregnancy)']

    X_train = train_scaled.drop(columns = drop_cols)
    y_train = train.heart_disease_Yes

    X_validate = validate_scaled.drop(columns = drop_cols)
    y_validate = validate.heart_disease_Yes

    X_test = test_scaled.drop(columns = drop_cols)
    y_test = test.heart_disease_Yes
    return X_train, y_train, X_validate, y_validate, X_test, y_test

############################ RANDOM FOREST ############################

def random_forest_model(X_train, X_validate, y_train, y_validate):
    # Evaluate Random Forest models on train & validate set by looping through different values for max_depth and min_samples_leaf hyperparameters

    # create empty list for which to append metrics from each loop
    scores = []
    # set value for range
    max_value = range(1,15) 

    # create loop for range 1-20
    for i in max_value:
        # set depth & n_samples to value for current loop
        depth = i
        n_samples = i

        # define the model setting hyperparameters to values for current loop
        forest = RandomForestClassifier(max_depth=depth, min_samples_leaf=n_samples, random_state=123)  

        # fit the model on train
        forest = forest.fit(X_train, y_train)   

        # use the model and evaluate performance on train
        in_sample_accuracy = forest.score(X_train, y_train)
        # use the model and evaluate performance on validate
        out_of_sample_accuracy = forest.score(X_validate, y_validate)

        # create output of current loop's hyperparameters and accuracy to append to metrics
        output = {
            "min_samples_per_leaf": n_samples,
            "max_depth": depth,
            "train_accuracy": in_sample_accuracy,
            "validate_accuracy": out_of_sample_accuracy
        }

        scores.append(output)   

    # convert metrics list to a dataframe for easy reading   
    df = pd.DataFrame(scores)
    # add column to assess the difference between train & validate accuracy
    df["difference"] = df.train_accuracy - df.validate_accuracy
    return df

############################ RANDOM FOREST REPORT ############################

def rf_report(X_train, X_validate, y_train, y_validate):
        # Evaluate Random Forest model on train & validate set

    # define the model setting hyperparameters to values for current loop
    forest = RandomForestClassifier(max_depth=9, min_samples_leaf=9, random_state=123)

    # fit the model on train
    forest = forest.fit(X_train, y_train)

    # use the model and evaluate performance on train
    train_accuracy = forest.score(X_train, y_train)
    # use the model and evaluate performance on validate
    validate_accuracy = forest.score(X_validate, y_validate)

    print(f'train_accuracy: {train_accuracy}')
    print(f'validate_accuracy: {validate_accuracy}')

############################ KNN ############################

    # Evaluate KNearest Neighbors models on train & validate set by looping through different values for k hyperparameter
def KNN_model(X_train, X_validate, y_train, y_validate):
    # create empty list for which to append scores from each loop
    scores = []
    k_range = range (1,20)
    # create loop for range 1-20
    for k in k_range:

        # define the model setting hyperparameters to values for current loop
        knn = KNeighborsClassifier(n_neighbors=k)

        # fit the model on train
        knn.fit(X_train, y_train)

        # use the model and evaluate performance on train
        train_accuracy = knn.score(X_train, y_train)
        # use the model and evaluate performance on validate
        validate_accuracy = knn.score(X_validate, y_validate)

        # create output of current loop's hyperparameters and accuracy to append to metrics
        output = {
            "k": k,
            "train_accuracy": train_accuracy,
            "validate_accuracy": validate_accuracy
        }

        scores.append(output)

    # convert scores list to a dataframe for easy reading
    df = pd.DataFrame(scores)
    # add column to assess the difference between train & validate accuracy
    df['difference'] = df.train_accuracy - df.validate_accuracy
    return df

############################ KNN RESULTS ############################

# Evaluate KNearest Neighbors model on train & validate dataset
def knn_results(X_train, y_train, X_validate, y_validate):           
    # define the model setting hyperparameter to 12
    knn = KNeighborsClassifier(n_neighbors=12)

    # fit the model on train
    knn.fit(X_train, y_train)

    # use the model and evaluate performance on train
    train_accuracy = knn.score(X_train, y_train)
    # use the model and evaluate performance on validate
    validate_accuracy = knn.score(X_validate, y_validate)

    print(f'train_accuracy: {train_accuracy: .2%}')
    print(f'validate_accuracy: {validate_accuracy: .2%}')

############################ LOGISTIC REGRESSION ############################

    # Evaluate Logistic Regression models on train & validate set by looping through different values for c hyperparameter
def logistic_regression_model(X_train, X_validate, y_train, y_validate):
    # create empty list for which to append metrics from each loop
    metrics = []

    # create loop for values in list
    for c in [.001, .005, .01, .05, .1, .5, 1, 5, 10, 50, 100, 500, 1000]:

        # define the model setting hyperparameters to values for current loop
        logit = LogisticRegression(C=c)

        # fit the model on train
        logit.fit(X_train, y_train)

        # use the model and evaluate performance on train
        train_accuracy = logit.score(X_train, y_train)
        # use the model and evaluate performance on validate
        validate_accuracy = logit.score(X_validate, y_validate)

        # create output of current loop's hyperparameters and accuracy to append to metrics
        output = {
            'C': c,
            'train_accuracy': train_accuracy,
            'validate_accuracy': validate_accuracy
        }

        metrics.append(output)

    # convert metrics list to a dataframe for easy reading
    df = pd.DataFrame(metrics)
    # add column to assess the difference between train & validate accuracy
    df['difference'] = df.train_accuracy - df.validate_accuracy
    return df

############################ LOGISTIC REGRESSION RESULTS ############################

def lr_results(X_train, y_train, X_validate, y_validate):
   # Evaluate Logistic Regession model on train and validate dataset

    logit = LogisticRegression(C=2)

    # fit the model on train
    logit.fit(X_train, y_train)

    # use the model and evaluate performance on train
    train_accuracy = logit.score(X_train, y_train)
    # use the model and evaluate performance on validate
    validate_accuracy = logit.score(X_validate, y_validate)

    print(f'train_accuracy: {train_accuracy: .2%}')
    print(f'validate_accuracy: {validate_accuracy: .2%}')

############################ MODEL ACCURACY ############################

def model_accuracy(X_train, y_train, X_validate, y_validate, X_test, y_test):
    #Define features used for the model
    x_cols =['age_category', 'diff_walking', 'diabetic_yes', 'physical_health', 'stroke_yes', 'smoking_yes', 'kidney_disease_Yes', 'gen_health']
    #Create Logistic Regression Model
    logit = LogisticRegression(random_state=123)
    # Fit the model
    logit.fit(X_train, y_train)
    # Establish weights
    weights = logit.coef_.flatten()
    # Establish intercept
    pd.DataFrame(weights, x_cols).reset_index().rename(columns={'index': 'x_cols', 0: 'weight'})
    logit = LogisticRegression(C=1, random_state=123)
    logit.fit(X_train, y_train)
    # Create a Random Forest model and set the number of trees and the max depth of 6 
    # based on loop used to find best performing k-value
    # Create the model with max depth of 16
    rf = RandomForestClassifier(max_depth=9,min_samples_leaf=9,random_state=123)
    # Fit the model
    rf.fit(X_train, y_train)  
    # Create a KNN model and set the number of neighbors to be used at 5
    knn = KNeighborsClassifier(n_neighbors=12)
    # Fit the model
    knn.fit(X_train,y_train)
    # Print the accuracy of each model
    print('====================================================================')
    # Accuracy on train for  Logistic Regression:
    print(f'Accuracy of Logistic Regression on the training set is {(logit.score(X_train, y_train)):.2%}')
    # Accurcy on validate for Logistic Regression:
    print(f'Accuracy of Logistic Regression on the validation set is {(logit.score(X_validate, y_validate)):.2%}')
    print('--------------------------------------------------------------------')
    # Accuracy on train for the Random Forest:
    print(f'Accuracy of Random Forest on the training set is {(rf.score(X_train, y_train)):.2%}')
    # Accurcy on validate for the Random Forest:
    print(f'Accuracy of Random Forest on the validation set is {(rf.score(X_validate, y_validate)):.2%}')
    print('--------------------------------------------------------------------')
    # Accuracy on train for  KNN:
    print(f'Accuracy of KNN on the training set is {(knn.score(X_train, y_train)):.2%}')
    # Accurcy on validate for KNN:
    print(f'Accuracy of KNN on the validation set is {(knn.score(X_validate, y_validate)):.2%}')
    print('====================================================================')
    return 

############################ KNN TEST ############################

def KNN_test(X_test, y_test, df):
    #Create the model
    model =  KNeighborsClassifier(n_neighbors=12)
    # Fit the model
    model.fit(X_test, y_test)
    # Evaluate the model
    # Accuracy on train for the Decision Tree:
    df['baseline'] = 0
    baseline_accuracy = (df.baseline == df.heart_disease_Yes).mean()
    print('==================================================================')
    print(f'Baseline accuracy for the training data set is {(baseline_accuracy):.2%}')
    # Accuracy on validate for the KNN:
    print(f'Accuracy of KNN on the test set is {(model.score(X_test, y_test)):.2%}')
    print('------------------------------------------------------------------')
    # By how much
    print(f'Accuracy gained with use of the new model on the test set is {(model.score(X_test, y_test) - baseline_accuracy):.2%}') 
    print('==================================================================')

############################ RANDOM FOREST TEST ############################

    # Evaluate Random Forest model on test dataset
def rf_test(X_test, y_test, test):    
    # define the model setting hyperparameters to values for current loop
    forest = RandomForestClassifier(max_depth=9, min_samples_leaf=9, random_state=123)

    test['baseline'] = 0
    baseline_accuracy = (test.baseline == test.heart_disease_Yes).mean()
    # fit the model on test
    forest = forest.fit(X_test, y_test)
    print('==================================================================')
    print(f'Baseline accuracy for the training data set is {(baseline_accuracy):.2%}')
     # Accuracy on validate for the KNN:
    print(f'Accuracy of Random Forest on the test set is {(forest.score(X_test, y_test)):.2%}')
    print('------------------------------------------------------------------')
    # By how much
    print(f'Accuracy gained with use of the new model on the test set is {(forest.score(X_test, y_test) - baseline_accuracy):.2%}')
    print('==================================================================')