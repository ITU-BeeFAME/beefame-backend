# app/service/evaluation_service.py
from service.utils.dataset_utils import calculate_average_odds_difference, calculate_disparate_impact, calculate_equal_opportunity_difference, calculate_statistical_parity_difference, calculate_theil_index
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from .utils.evaluation_utils import run_data_repairer, run_label_flipping, run_prevalence_sampling, run_relabeller
from ucimlrepo import fetch_ucirepo

class EvaluationService:
    def __init__(self): 
        pass

    def evaluate(self, dataset_list, classifier_list, method_list):
        """ TO DO : Evaluation must be applied for multiple items """
        print("datasetlist", dataset_list)
        print("classifier_list", classifier_list)
        print("method_list", method_list)
        test_size = 0.2
        random_state = 42
        datasets = {144 : "german", 2 : "adult"}

        classifiers = {
            "Logistic Regression": LogisticRegression(max_iter=2000, random_state=random_state),
            "SVM": SVC(probability=True),
            "Random Forest": RandomForestClassifier(random_state=random_state),
            "XGBClassifier": XGBClassifier(random_state=random_state)
        }

        methods = {
            "Label Flipping": run_label_flipping,
            "Data Repairer": run_data_repairer,
            "Prevalence Sampling": run_prevalence_sampling,
            "Relabeller": run_relabeller
        }

        if "german" not in dataset_list: datasets.pop(144)
        if "adult" not in dataset_list: datasets.pop(2)

        if "Logistic Regression" not in classifier_list: classifiers.pop("Logistic Regression")
        if "SVM" not in classifier_list: classifiers.pop("SVM")
        if "Random Forest" not in classifier_list: classifiers.pop("Random Forest")
        if "XGBClassifier" not in classifier_list: classifiers.pop("XGBClassifier")

        if "Label Flipping" not in method_list: methods.pop("Label Flipping")
        if "Data Repairer" not in method_list: methods.pop("Data Repairer")
        if "Prevalence Sampling" not in method_list: methods.pop("Prevalence Sampling")
        if "Relabeller" not in method_list: methods.pop("Relabeller")

        final_metrics = []

        for dataset_id, dataset_name in datasets.items():
            dataset = fetch_ucirepo(id=dataset_id)

            X = dataset.data.features
            y = dataset.data.targets

            if isinstance(y, pd.DataFrame):
                y = y.squeeze()  # Convert DataFrame to Series if necessary

            if dataset_id == 2:
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
            elif dataset_id == 144:
                if 'Attribute13' in X.columns:
                    age_threshold = 50
                    X['age_binary'] = (X['Attribute13'] >= age_threshold).astype(int)
                    X.drop('Attribute13', axis=1, inplace=True)  # Remove the original 'age' column
                sensitive_columns = ['Attribute9_A91', 'age_binary']
                sensitive_columns_display = {'Attribute9_A91': 'Gender','age_binary' : 'Age'}
            
            # Handling potential SettingWithCopyWarning correctly
            X = X.copy().replace('?', np.nan).dropna()
            y = y.loc[X.index]

            # Ensure y is binary (0 and 1)
            if y.nunique() == 2 and set(y.unique()).issubset({1, 2}):
                # Map 1 -> 0 and 2 -> 1
                y = y.replace({1: 0, 2: 1}).astype(int)
            
            # One-hot encode categorical variables
            X = pd.get_dummies(X)

            protected_attribute = pd.Series(X[sensitive_columns[0]].values, dtype=int)

            # Split data
            X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
                X, y, protected_attribute, test_size=test_size, random_state=random_state)
            
            # Ensure all feature names are strings
            X_train.columns = X_train.columns.astype(str)
            X_test.columns = X_test.columns.astype(str)

            s_train = s_train.astype('category')

            # Standardize the data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train and evaluate models for each method and classifier
            for sensitive_column in sensitive_columns:
                for method_name, method in methods.items():
                    print(f"Processing: {method_name}")

                    # Reload the original X and y data for each method
                    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(X, y, protected_attribute, test_size=test_size, random_state=random_state)

                    # Ensure all feature names are strings
                    X_train.columns = X_train.columns.astype(str)
                    X_test.columns = X_test.columns.astype(str)

                    s_train = s_train.astype('category')
                    
                    # Apply the method
                    X_train_transformed, y_train_transformed = method(X_train, y_train, s_train)

                    X_train_transformed_scaled = scaler.fit_transform(X_train_transformed)
                    X_test_scaled = scaler.transform(X_test)  # Ensure the test data is scaled with the same scaler

                    # Explicitly add column names to y_test and s_test
                    y_test = pd.DataFrame(y_test.values, columns=['class'])  # Ensure y_test has a 'class' column
                    s_test = pd.DataFrame(s_test.values, columns=[sensitive_column])

                    for model_name, model in classifiers.items():
                        model.fit(X_train_transformed_scaled, y_train_transformed)
                        y_pred = model.predict(X_test_scaled)

                        accuracy = accuracy_score(y_test, y_pred)

                        final_metrics.append({
                                "Sensitive Column" : sensitive_columns_display[sensitive_column],
                                "Dataset Name" : dataset_name,
                                "Method Name" : method_name,
                                "Model Name" : model_name,
                                "Model Accuracy" : accuracy,
                                "Statistical Parity Difference" : calculate_statistical_parity_difference(X_test, y_test, y_pred, sensitive_column),
                                "Equal Opportunity Difference" : calculate_equal_opportunity_difference(X_test, y_test, y_pred, sensitive_column),
                                "Average Odds Difference" : calculate_average_odds_difference(X_test, y_test, y_pred, sensitive_column),
                                "Disparate Impact" : calculate_disparate_impact(X_test, y_test, y_pred, sensitive_column),
                                "Theil Index" : calculate_theil_index(y_test, y_pred)
                        })
        return final_metrics

            