# app/service/evaluation_service.py
from model.classifier import ClassifierName
from model.dataset import DatasetName
from model.method import MethodName
from db.firebaseConfig import FirebaseConfig
from typing import List, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from themis_ml.datasets import german_credit, census_income
from .utils.evaluation_utils import run_data_repairer, run_label_flipping, run_prevalence_sampling, run_relabeller
from model.evaluation import ClassificationReport, EvaluationResult

class EvaluationService:
    def __init__(self): 
        firebase_config = FirebaseConfig()
        self.db = firebase_config.get_db()

    def get_evaluation(dataset_name: DatasetName, classifier_name: ClassifierName, method_name: MethodName) -> EvaluationResult:
        # Dataset loading based on dataset_name
        if dataset_name == DatasetName.GERMAN:
            df = german_credit(raw=True)
            label_column = 'credit_risk'
            privileged_groups = ["telephone", "foreign_worker"]
            selected_privileged_group = "telephone"
        else:  # ADULT
            df = census_income(raw=True)
            label_column = 'income'
            privileged_groups = ["race", "sex"]
            selected_privileged_group = "sex"

        # Parameters
        test_size = 0.2
        random_state = 82

        # Prepare labels, features, and protected attribute
        labels = pd.Series(df[label_column].values)
        features = pd.get_dummies(df.drop(label_column, axis=1))

        # Ensure all boolean columns are converted to integers
        for col in features.columns:
            if features[col].dtype == 'bool':
                features[col] = features[col].astype(int)

        protected_attribute = pd.Series(df[selected_privileged_group].values, dtype=int)

        # Split data
        X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
            features, labels, protected_attribute, test_size=test_size, random_state=random_state)

        # Ensure all feature names are strings
        X_train.columns = X_train.columns.astype(str)
        X_test.columns = X_test.columns.astype(str)

        s_train = s_train.astype('category')

        # Standardize the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Select method based on method_name
        methods = {
            MethodName.DataRepairer: run_data_repairer,
            MethodName.PrevalanceSampling: run_prevalence_sampling,
            MethodName.Relabeller: run_relabeller
        }
        
        selected_method = methods[method_name]

        # Select classifier based on classifier_name
        classifiers = {
            ClassifierName.XGB: XGBClassifier(random_state=random_state),
            ClassifierName.SVC: SVC(probability=True),
            ClassifierName.RFC: RandomForestClassifier(random_state=random_state),
            ClassifierName.LR: LogisticRegression(max_iter=2000, random_state=random_state)
        }
        
        selected_classifier = classifiers[classifier_name]

        # Apply the selected method
        X_train_transformed, y_train_transformed = selected_method(X_train, y_train, s_train)

        X_train_transformed_scaled = scaler.fit_transform(X_train_transformed)
        X_test_scaled = scaler.transform(X_test)

        # Train and evaluate the selected classifier
        selected_classifier.fit(X_train_transformed_scaled, y_train_transformed)
        predictions = selected_classifier.predict(X_test_scaled)
        prob_predictions = (selected_classifier.predict_proba(X_test_scaled)[:, 1] 
                        if hasattr(selected_classifier, "predict_proba") 
                        else selected_classifier.decision_function(X_test_scaled))

        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        balanced_acc = balanced_accuracy_score(y_test, predictions)
        auc_roc = roc_auc_score(y_test, prob_predictions)
        report_dict = classification_report(y_test, predictions, output_dict=True)
        conf_matrix = confusion_matrix(y_test, predictions)

        # Create datasets for fairness metrics
        dataset_true = BinaryLabelDataset(
            df=pd.concat([
                X_test.reset_index(drop=True),
                pd.DataFrame(y_test.values, columns=[label_column]),
                pd.DataFrame(s_test.values, columns=[selected_privileged_group])
            ], axis=1),
            label_names=[label_column],
            protected_attribute_names=[selected_privileged_group]
        )

        dataset_pred = dataset_true.copy()
        dataset_pred.labels = predictions.reshape(-1, 1)
        
        metric = ClassificationMetric(
            dataset_true, 
            dataset_pred,
            unprivileged_groups=[{selected_privileged_group: 0}],
            privileged_groups=[{selected_privileged_group: 1}]
        )

        # Prepare results
        report_df = pd.DataFrame(report_dict).transpose()
        
        classification_result = ClassificationReport(
            precision=report_df.get('precision', {}).to_dict(),
            recall=report_df.get('recall', {}).to_dict(),
            f1_score=report_df.get('f1-score', {}).to_dict(),
            support=report_df.get('support', {}).to_dict()
        )

        evaluation_result = EvaluationResult(
            name=f"{method_name.value} - {classifier_name.value}",
            accuracy=accuracy,
            balanced_accuracy=balanced_acc,
            auc_roc=auc_roc,
            classification_report=classification_result,
            disparate_impact=metric.disparate_impact(),
            statistical_parity_difference=metric.statistical_parity_difference(),
            equal_opportunity_difference=metric.equal_opportunity_difference(),
            average_odds_difference=metric.average_odds_difference(),
            theil_index=metric.theil_index()
        )

        return evaluation_result

    def get_evaluation(self) -> List[EvaluationResult]:
        german_credit_df = german_credit(raw=True)
        census_income_df = census_income(raw=True)  # Adult dataset

        # Parameters
        test_size = 0.2
        random_state = 82
        privileged_groups1 = "foreign_worker"
        privileged_groups2 = "telephone"
        selected_privileged_groups = privileged_groups2

        # Prepare labels, features, and protected attribute
        labels = pd.Series(german_credit_df['credit_risk'].values)
        features = pd.get_dummies(german_credit_df.drop('credit_risk', axis=1))

        # Ensure all boolean columns are converted to integers
        for col in features.columns:
            if features[col].dtype == 'bool':
                features[col] = features[col].astype(int)

        protected_attribute = pd.Series(german_credit_df[selected_privileged_groups].values, dtype=int)

        # Split data
        X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
            features, labels, protected_attribute, test_size=test_size, random_state=random_state)

        # Ensure all feature names are strings
        X_train.columns = X_train.columns.astype(str)
        X_test.columns = X_test.columns.astype(str)

        s_train = s_train.astype('category')

        # Standardize the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        methods = {
            # "Label Flipping": run_label_flipping,
            "Data Repairer": run_data_repairer,
            "Prevalence Sampling": run_prevalence_sampling,
            "Relabeller": run_relabeller
        }

        results = {}

        # Train and evaluate models for each method and classifier
        for name, method in methods.items():
            print(f"Processing: {name}")

            # Reload the original X and y data for each method
            X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
                features, labels, protected_attribute, test_size=test_size, random_state=random_state)

            # Ensure all feature names are strings
            X_train.columns = X_train.columns.astype(str)
            X_test.columns = X_test.columns.astype(str)

            s_train = s_train.astype('category')

            # Apply the method
            X_train_transformed, y_train_transformed = method(X_train, y_train, s_train)

            X_train_transformed_scaled = scaler.fit_transform(X_train_transformed)
            X_test_scaled = scaler.transform(X_test)  # Ensure the test data is scaled with the same scaler

            classifiers = {
                "SVM": SVC(probability=True),
                "Random Forest": RandomForestClassifier(random_state=random_state),
                "Logistic Regression": LogisticRegression(max_iter=2000, random_state=random_state),
                "XGBClassifier": XGBClassifier(random_state=random_state)
            }

            for clf_name, clf in classifiers.items():
                clf.fit(X_train_transformed_scaled, y_train_transformed)
                predictions = clf.predict(X_test_scaled)
                prob_predictions = clf.predict_proba(X_test_scaled)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_test_scaled)

                accuracy = accuracy_score(y_test, predictions)
                balanced_acc = balanced_accuracy_score(y_test, predictions)
                auc_roc = roc_auc_score(y_test, prob_predictions)
                report_dict = classification_report(y_test, predictions, output_dict=True)
                conf_matrix = confusion_matrix(y_test, predictions)

                dataset_true = BinaryLabelDataset(df=pd.concat([X_test.reset_index(drop=True),
                                                                pd.DataFrame(y_test.values, columns=['credit_risk']),
                                                                pd.DataFrame(s_test.values, columns=[selected_privileged_groups])], axis=1),
                                                label_names=['credit_risk'],
                                                protected_attribute_names=[selected_privileged_groups])

                dataset_pred = dataset_true.copy()
                dataset_pred.labels = predictions.reshape(-1, 1)
                metric = ClassificationMetric(dataset_true, dataset_pred,
                                            unprivileged_groups=[{selected_privileged_groups: 0}],
                                            privileged_groups=[{selected_privileged_groups: 1}])

                # Store results
                results[f"{name} - {clf_name}"] = {
                    "accuracy": accuracy,
                    "balanced_accuracy": balanced_acc,
                    "auc_roc": auc_roc,
                    "report": report_dict,
                    "disparate_impact": metric.disparate_impact(),
                    "statistical_parity_difference": metric.statistical_parity_difference(),
                    "equal_opportunity_difference": metric.equal_opportunity_difference(),
                    "average_odds_difference": metric.average_odds_difference(),
                    "theil_index": metric.theil_index(),
                    "conf_matrix": conf_matrix
                }

        all_results = []

        for name, result in results.items():
            report_df = pd.DataFrame(result['report']).transpose()

            precision = report_df.get('precision', {}).to_dict()
            recall = report_df.get('recall', {}).to_dict()
            f1_score = report_df.get('f1-score', {}).to_dict()
            support = report_df.get('support', {}).to_dict()

            classification_result = ClassificationReport(
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                support=support
            )

            evaluation_result = EvaluationResult(
                name=name,
                accuracy=result['accuracy'],
                balanced_accuracy=result['balanced_accuracy'],
                auc_roc=result['auc_roc'],
                classification_report=classification_result,
                disparate_impact=result['disparate_impact'],
                statistical_parity_difference=result['statistical_parity_difference'],
                equal_opportunity_difference=result['equal_opportunity_difference'],
                average_odds_difference=result['average_odds_difference'],
                theil_index=result['theil_index']
            )

            all_results.append(evaluation_result)

        return all_results