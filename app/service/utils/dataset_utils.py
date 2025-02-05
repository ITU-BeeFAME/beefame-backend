from typing import List
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    true_positive_rate,
    false_positive_rate
)
import pandas as pd

from model.classifier import ClassifierName
from model.dataset import DatasetName
pd.set_option('future.no_silent_downcasting', True)
import numpy as np
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from ucimlrepo import fetch_ucirepo

def calculate_statistical_parity_difference(X, y_true, y_pred, sensitive_column):
    sr = lambda y_true, y_pred: selection_rate(y_true, y_pred)
    mf = MetricFrame(metrics=sr, y_true=y_true, y_pred=y_pred, sensitive_features=X[sensitive_column])
    return mf.difference(method='between_groups')

def calculate_equal_opportunity_difference(X, y_true, y_pred, sensitive_column, pos_label=1):
    tpr = lambda y_true, y_pred: true_positive_rate(y_true, y_pred, pos_label=pos_label)
    mf = MetricFrame(metrics=tpr, y_true=y_true, y_pred=y_pred, sensitive_features=X[sensitive_column])
    return mf.difference(method='between_groups')

def calculate_average_odds_difference(X, y_true, y_pred, sensitive_column):
    unique_labels = np.unique(y_true)
    if len(unique_labels) == 2:
        pos_label = unique_labels[1] 
    else:
        raise ValueError("y_true should have exactly two unique values for binary classification")

    tpr = lambda y_true, y_pred: true_positive_rate(y_true, y_pred, pos_label=pos_label)
    fpr = lambda y_true, y_pred: false_positive_rate(y_true, y_pred, pos_label=pos_label)
    average_odds = lambda y_true, y_pred: (tpr(y_true, y_pred) + fpr(y_true, y_pred)) / 2

    mf = MetricFrame(metrics=average_odds, 
                     y_true=y_true, 
                     y_pred=y_pred, 
                     sensitive_features=X[sensitive_column])
    return mf.difference(method='between_groups')

def calculate_disparate_impact(X, y_true, y_pred, sensitive_column):
    sr = lambda y_true, y_pred: selection_rate(y_true, y_pred)
    mf = MetricFrame(metrics=sr, y_true=y_true, y_pred=y_pred, sensitive_features=X[sensitive_column])
    return mf.ratio(method='between_groups')

def calculate_theil_index(y_true, y_pred):
    actual_pos = np.mean(y_true == 1)
    pred_pos = np.mean(y_pred == 1)

    epsilon = 1e-10

    actual_entropy = -(actual_pos * np.log2(actual_pos + epsilon) + (1 - actual_pos) * np.log2(1 - actual_pos + epsilon))
    pred_entropy = -(pred_pos * np.log2(pred_pos + epsilon) + (1 - pred_pos) * np.log2(1 - pred_pos + epsilon))

    theil_index = pred_entropy - actual_entropy
    return theil_index

def initial_dataset_analysis(dataset_names: List[str], classifiers: List[str]):
    print("datasetname", dataset_names)
    print("classifier", classifiers)
   
    dataset_name = dataset_names[0].value
    classifier = classifiers[0].value

    """ TO DO : Analysis must be applied for multiple items """
    dataset_dict = {"german" : 144, "adult" : 2}

    dataset = fetch_ucirepo(id=dataset_dict[dataset_name]) 
    
    X = dataset.data.features 
    y = dataset.data.targets 

    if isinstance(y, pd.DataFrame):
        y = y.squeeze()  # Convert DataFrame to Series if necessary

    if dataset_name == "adult":
        y = y.replace({'<=50K': 0, '<=50K.': 0, '>50K': 1, '>50K.': 1}).astype(int)
        if 'age' in X.columns:
            age_threshold = 50
            X['age_binary'] = (X['age'] >= age_threshold).astype(int)
            X.drop('age', axis=1, inplace=True)  # Remove the original 'age' column
        if 'race' in X.columns:
            X['race_binary'] = (X['race'] == 'White').astype(int)
            X.drop('race', axis=1, inplace=True)  # Optionally remove the original 'race' column
        sensitive_columns = ['sex_Female', 'age_binary', 'race_binary']  
        sensitive_columns_display = {'sex_Female': 'Gender', 'age_binary': "Age", 'race_binary': "Race"}

    elif dataset_name == "german":
        if 'Attribute13' in X.columns:
            age_threshold = 50
            X['age_binary'] = (X['Attribute13'] >= age_threshold).astype(int)
            X.drop('Attribute13', axis=1, inplace=True)  # Remove the original 'age' column
        sensitive_columns = ['Attribute9_A91', 'age_binary'] 
        sensitive_columns_display = {'Attribute9_A91': 'Gender',
                                'age_binary' : 'Age'}

    # Handling potential SettingWithCopyWarning correctly
    X = X.copy().replace('?', np.nan).dropna()
    y = y.loc[X.index]

    """Bu kalacak"""
    if y.nunique() == 2 and set(y.unique()).issubset({1, 2}):
        # Map 1 -> 0 and 2 -> 1
        y = y.replace({1: 0, 2: 1}).astype(int)

    # One-hot encode categorical variables
    X = pd.get_dummies(X)
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling features only at the point of training
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # model default xgb
    model = LogisticRegression(random_state=42, max_iter=10000)

    # Model training
    if classifier == ClassifierName.SVC.value:
        model = SVC(probability=True)
    elif classifier == ClassifierName.RFC.value:
        model = RandomForestClassifier(random_state=42)
    elif classifier == ClassifierName.XGB.value:
        model = XGBClassifier(random_state=42)

    model.fit(X_train_scaled, y_train)

    # Model prediction
    y_pred = model.predict(X_test_scaled)

    # Model evaluation
    accuracy = accuracy_score(y_test, y_pred)

    final_metrics = []
    for sensitive_column in sensitive_columns:
        if sensitive_column in X.columns:
            final_metrics.append({
                "Sensitive Column" : sensitive_columns_display[sensitive_column],
                "Model Accuracy" : accuracy,
                "Statistical Parity Difference" : calculate_statistical_parity_difference(X_test, y_test, y_pred, sensitive_column),
                "Equal Opportunity Difference" : calculate_equal_opportunity_difference(X_test, y_test, y_pred, sensitive_column),
                "Average Odds Difference" : calculate_average_odds_difference(X_test, y_test, y_pred, sensitive_column),
                "Disparate Impact" : calculate_disparate_impact(X_test, y_test, y_pred, sensitive_column),
                "Theil Index" : calculate_theil_index(y_test, y_pred)
            })

    return final_metrics

