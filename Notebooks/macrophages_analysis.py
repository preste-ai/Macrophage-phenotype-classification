import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scikitplot as skplt
import re
from xgboost import XGBClassifier
from itertools import cycle
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler, label_binarize, LabelEncoder
from sklearn.metrics import roc_curve, auc, RocCurveDisplay, confusion_matrix
from sklearn.utils import class_weight


def make_nucleus_classifier(nucleus_dataset, confusion_matrix=False, no_NT=False):
    """
    Create and train an XGBoost classifier for the nucleus dataset.

    Args:
        nucleus_dataset (DataFrame): Input dataset containing features and labels.

    Returns:
        tuple: A tuple containing a confusion matrix, trained classifier, and test data.
    """
    # Create an XGBoost classifier with specified hyperparameters
    classifier = XGBClassifier(colsample_bytree=0.9,
                              gamma=0.2,
                              learning_rate=0.1,
                              max_depth=5,
                              min_child_weight=1,
                              n_estimators=200,
                              subsample=0.8,
                              use_label_encoder=False,
                              eval_metric='mlogloss')
    
    # Create a pipeline that includes data preprocessing steps and the classifier
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('classifier', classifier)])
    
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    
    # Separate features (X_nucleus) and target labels (y_nucleus)
    X_nucleus = nucleus_dataset.drop('CellType', axis=1)
    y_nucleus = nucleus_dataset['CellType']
    
    # Encode labels in column 'CellType'
    label_encoder = LabelEncoder()
    y_nucleus = label_encoder.fit_transform(y_nucleus)
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_nucleus, y_nucleus, test_size=0.2, random_state=0, stratify=y_nucleus)
    
    classes_weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_train)
    
    # Fit the pipeline to the training data and make a prediction
    pipe.fit(X_train, y_train, classifier__sample_weight=classes_weights)
    y_pred = pipe.predict(X_test)
    
    # Calculate accuracy and f1-macro scores using cross-validation
    acc_scores = cross_val_score(pipe, X_test, y_test, cv=kf, scoring='accuracy')
    f1_macro = cross_val_score(pipe, X_test, y_test, cv=kf,scoring='f1_macro')
    f1_weighted = cross_val_score(pipe, X_test, y_test, cv=kf,scoring='f1_weighted')
    
    mean_accuracy = sum(acc_scores) / len(acc_scores)
    mean_f1_macro = sum(f1_macro) / len(f1_macro)
    mean_f1_weighted = sum(f1_weighted) / len(f1_weighted)
    
    print('Mean accuracy of XGBClassifier (10-fold cross-validation)', mean_accuracy)
    print('Mean f1-macro of XGBClassifier (10-fold cross-validation)', mean_f1_macro)
    print('Mean f1_weighted of XGBClassifier (10-fold cross-validation)', mean_f1_weighted)
    
    if (no_NT == True) and (confusion_matrix == True):
        skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=False, title='Confusion Matrix')
        plt.xticks(range(2),['M1', 'M2'])
        plt.yticks(range(2),['M1', 'M2'])
        plt.show()
    elif (no_NT == False) and (confusion_matrix == True):
        skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=False, title='Confusion Matrix')
        plt.xticks(range(3),['M1', 'M2', 'NT'])
        plt.yticks(range(3),['M1', 'M2', 'NT'])
        plt.show()
        
    
    test_data = (X_test, y_test)

    
    return pipe, test_data

    
def make_multiclass_roc(pipe, test_data):
    """
    Create a multi-class ROC curve for a trained classifier.

    Args:
        nucleus_classifier: Trained classifier.
        test_data (tuple): Tuple containing test features and labels.

    Returns:
        None
    """
    X_test, y_test = test_data
    
    # Binarize the labels into a one-hot encoded format
    y_test = label_binarize(y_test, classes=[0, 1, 2])
    
    # Predict class probabilities using the classifier
    class_probabilities = pipe.predict_proba(X_test)

    # Initialize empty lists for storing ROC curve data for each class
    n_classes = 3
    fpr = [0] * 3
    tpr = [0] * 3
    thresholds = [0] * 3
    auc_score = [0] * 3
    
    precision_scores = [0] * 3
    recall_scores = [0] * 3
    f1_scores = [0] * 3

    # Calculate the ROC curve and AUC for each class
    for i in range(n_classes):
        fpr[i], tpr[i], thresholds[i] = roc_curve(y_test[:, i], class_probabilities[:, i])
        auc_score[i] = auc(fpr[i], tpr[i])

    # Create a figure and axis for plotting
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot the chance level line (AUC = 0.5)
    ax.plot([0, 1], [0, 1], "k--", label="Chance level (AUC = 0.5)")

    # Define target names and colors for plotting ROC curves
    target_names = ['M1', 'M2', 'NT']
    colors = cycle(["darkblue", "lightblue", "magenta"])
    
    # Plot ROC curves for each class
    for class_id, color in zip(range(3), colors):
        RocCurveDisplay.from_predictions(
            y_test[:, class_id],
            class_probabilities[:, class_id],
            name=f"ROC curve for {target_names[class_id]}",
            color=color,
            ax=ax)

    average_AUC = sum(auc_score) / n_classes
    
    print('AUC score for M1', auc_score[0])
    print('AUC score for M2', auc_score[1])
    print('AUC score for NT', auc_score[2])
    print('Average AUC', average_AUC)


    
def calculate_metrics_for_M1_M2(pipe, test_data):
    """
    Calculate performance metrics for a binary classification model on M1 and M2 classes.

    Args:
        pipe (sklearn.pipeline.Pipeline): Trained scikit-learn pipeline containing the classifier.
        test_data (tuple): A tuple containing test features and their corresponding labels.

    Prints:
        Confusion matrix for M1 and M2 classes.
        Accuracy, Precision, Sensitivity, Specificity, F1-score
    """
    y_pred = pipe.predict(test_data[0])
    confusion_matrix_m1m2 = confusion_matrix(test_data[1], y_pred)

    # Extract true positives, false positives, true negatives, and false negatives
    true_positives = confusion_matrix_m1m2[0][0]
    false_positives = confusion_matrix_m1m2[1][0]
    true_negatives = confusion_matrix_m1m2[1][1]
    false_negatives = confusion_matrix_m1m2[0][1]

    # Calculate metrics
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    specificity = true_negatives / (true_negatives + false_positives)
    f1 = 2 * (precision * recall) / (precision + recall)

    print('Calculate metrics for M1 and M2 only')
    print('____________________________________')
    print(' M1,   M2')
    print(confusion_matrix_m1m2[:2,:2])
    print("Accuracy (Correctly predicted M1+M2 / all predictions):", accuracy)
    print("Precision (Correctly predicted M1 / all cells predicted as M1):", precision)
    print("Sensitivity (Correctly predicted M1 / all actual M1):", recall)
    print("Specificity (Correctly predicted M2 / all actual M2):", specificity)
    print("F1-score (How good the model performs in correctly identifying M1 and avoiding misclassifying M2):", f1)


def get_best_features(pipe, feature_names, features_amount):
    """
    Get the top N important features from a trained classifier.

    Args:
        nucleus_classifier: Trained classifier.
        feature_names (DataFrame): Dataframe containing feature names.
        features_amount (int): Number of top features to select.

    Returns:
        DataFrame: Dataframe containing the top N important features.
    """
    
    # Get the trained nucleus classifier from the pipeline
    nucleus_classifier = pipe.steps[1][1]
    
    # Get feature importances from the trained classifier
    coefficients = nucleus_classifier.feature_importances_
    
    # Create a DataFrame to store feature names and their importance
    feature_importance = pd.DataFrame({'Feature': feature_names.columns, 'Importance': np.abs(coefficients)})
    
    # Sort the features by importance ang get top features
    feature_importance.sort_values('Importance', ascending=False, inplace=True)
    top_features = feature_importance.head(features_amount)

    return top_features


def remove_correlated_features(correlated_data, threshold):
    """
    Remove highly correlated numeric features from the input data while keeping all non-numeric columns.

    Args:
        correlated_data (DataFrame): Input data with potentially correlated features.
        threshold (float): Correlation threshold above which numeric features will be removed.

    Returns:
        DataFrame: Input data with highly correlated numeric features removed while keeping all non-numeric columns.
    """
    # Select numeric columns from the input data
    numeric_data = correlated_data.select_dtypes(include=[np.number])

    # Calculate the correlation matrix for the numeric data
    corr_matrix = numeric_data.corr().abs()

    # Create an upper triangular matrix to identify highly correlated features
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find the numeric columns (features) with correlation greater than the threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    # Drop the highly correlated numeric features from the input data
    print('Amount of correlated features to drop:', len(to_drop))
    non_correlated_data = correlated_data.drop(to_drop, axis=1)

    return non_correlated_data


def select_features_by_name(nucleus_dataset, pattern_to_select=''):
    """
    Remove features from a list of datasets based on a given feature name pattern.

    Args:
        nucleus_dataset: Pandas DataFrame representing datasets.
        pattern_to_select: A string pattern to identify features for selection (default is an empty string).
        
    Returns:
        A list of truncated datasets with features other than selected removed.
    """
    
    nucleus_features = nucleus_dataset.columns.to_list()
    features_to_select = []

    # find features that contain specified in the 'pattern_to_remove' pattern
    for feature in nucleus_features:
        if re.search(pattern_to_select, feature) or feature == 'CellType':
            # Add the feature to the list for removal
            features_to_select.append(feature)
        
    return nucleus_dataset.loc[:, features_to_select]


def remove_features_by_name(nucleus_datasets, pattern_to_remove=''):
    """
    Remove features from a list of datasets based on a given feature name pattern.

    Args:
        nucleus_datasets: A list of Pandas DataFrames representing datasets.
        pattern_to_remove: A string pattern to identify features for removal (default is an empty string).
        
    Returns:
        A list of truncated datasets with specified features removed.
    """
    truncated_datasets = []
    
    for dataset in nucleus_datasets:
        nucleus_features = dataset.columns.to_list()
        features_to_remove = []

        # find features that contain specified in the 'pattern_to_remove' pattern
        for feature in nucleus_features:
            if not re.search(pattern_to_remove, feature):
                # Add the feature to the list for removal
                features_to_remove.append(feature)
                
        truncated_datasets.append(dataset.loc[:, features_to_remove])  # Append the truncated dataset to the result list
        
    return truncated_datasets


def remove_outliers(dataframe, colname, method='percentile'):
    """
    Remove outliers from a DataFrame column using the IQR method.

    Args:
        dataframe (DataFrame): The input DataFrame containing the data.
        colname (str): The name of the column from which outliers will be removed.

    Returns:
        DataFrame: A new DataFrame with outliers removed.
    """
    if method == 'percentile':
        lower_bound = dataframe[colname].quantile(0.01)
        upper_bound = dataframe[colname].quantile(0.99)
        
    elif method == 'IQR':
        # Calculate 1st and 3rd quartiles
        Q1 = dataframe[colname].quantile(0.25)
        Q3 = dataframe[colname].quantile(0.75)
        IQR = Q3 - Q1

        # Calculate the lower and upper bound for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
    
    # Find the indices of outliers based on the bounds
    index_outlier = dataframe[(dataframe[colname] < lower_bound) | (dataframe[colname] > upper_bound)].index
    
    # Remove outliers and return the new DataFrame
    return dataframe.drop(index_outlier)


def write_features(features_list, file_name):
    """
    Write a list of features into a text file.

    Args:
        features_list (list): list of selected features.
        file_name (str): Name of the file to read features from.

    Returns:
        list: List of features read from the file.
    """
    with open(file_name, 'w') as file:
        file.writelines(line+'\n' for line in features_list)
        print(f'Features are written to {file_name}')
    
    
def read_features(file_name):
    """
    Read a list of features from a text file.

    Args:
        file_name (str): Name of the file to read features from.

    Returns:
        list: List of features read from the file.
    """
    features_list = []
    
    with open(file_name, 'r') as file:
        features_list = [line.strip() for line in file.readlines()]

    return features_list


def perform_undersampling(dataset):
    """
    Perform undersampling to balance the dataset based on the minority class.

    Args:
        dataset (DataFrame): The input dataset containing class labels.

    Returns:
        DataFrame: A new DataFrame with balanced class distribution through undersampling.
    """
    # Separate data for each class: 'M1', 'M2', and 'NT'
    m1 = dataset[dataset['CellType'] == 'M1']
    m2 = dataset[dataset['CellType'] == 'M2']
    nt = dataset[dataset['CellType'] == 'NT']
    
    # Determine the size of the minority class
    min_class_size = min([len(m1), len(m2), len(nt)])
    print('Balancing data by minority class. Minority class size is:', min_class_size)
    
    # Sample a subset of data from each class to match the size of the minority class
    m1_undersampled = m1.sample(min_class_size)
    m2_undersampled = m2.sample(min_class_size)
    nt_undersampled = nt.sample(min_class_size)
    
    # Concatenate the undersampled data to create a balanced dataset
    undersampled_dataset = pd.concat([m1_undersampled, m2_undersampled, nt_undersampled], axis=0)
    
    return undersampled_dataset