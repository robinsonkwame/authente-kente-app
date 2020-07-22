from abc import ABC, abstractmethod

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegressionCV


class TrainingModel(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def train(self, X_embedded, X_val_embedded, y_train, y_val):
        pass

    def fit_eval(self, search_algorithm, X_embedded, X_val_embedded, y_train, y_val):
        search_algorithm.fit(X_embedded,
                   y_train)
        best_estimator = search_algorithm.best_estimator_

        preds = best_estimator.predict(X_val_embedded)
        avg_precision = average_precision_score(y_val, preds)
        report = classification_report(y_val, preds, output_dict=True)

        return avg_precision, report, best_estimator


class LOFTrainer(TrainingModel):
    def __init__(self):
        super().__init__()

    def train(self, X_embedded, X_val_embedded, y_train, y_val):
        n_neighbors = {"n_neighbors": [1,5,11]}
        metric = {"metric": ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'] +['correlation', 'seuclidean', 'sqeuclidean']}
        novelty = {"novelty":[True]}

        parameter_grid = {**n_neighbors,
                          **metric,
                          **novelty}
        folds = StratifiedKFold(n_splits=3).split(X_embedded, y_train)
        search = GridSearchCV(
            estimator=LocalOutlierFactor(),
            param_grid=parameter_grid,
            scoring=('f1_macro'),
            cv=folds,
            verbose=5,
            n_jobs=-1,
            )

        avg_precision, report, best_estimator = super().fit_eval(search,
                                    X_embedded, X_val_embedded, y_train, y_val)

        return avg_precision, report, best_estimator



class IsolationForestTrainer(TrainingModel):
    def __init__(self):
        super().__init__()

    def train(self, X_embedded, X_val_embedded, y_train, y_val):
        n_estimators = {"n_estimators": [50,200,11]}
        n_jobs = {"n_jobs":[-1]}
        parameter_grid = {**n_estimators,
                          **n_jobs}

        folds = StratifiedKFold(n_splits=3).split(X_embedded, y_train)
        search = GridSearchCV(
            estimator=IsolationForest(),
            param_grid=parameter_grid,
            scoring=('f1_macro'),
            cv=folds,
            verbose=5,
            n_jobs=-1,
            )

        avg_precision, report, best_estimator = super().fit_eval(search,
                                    X_embedded, X_val_embedded, y_train, y_val)

        return avg_precision, report, best_estimator


class LogisticRegressionTrainer(TrainingModel):
    def __init__(self):
        super().__init__()

    def train(self, X_embedded, X_val_embedded, y_train, y_val):
        regressor = LogisticRegressionCV(n_jobs=-1,
                                         cv=3,
                                         scoring='f1_macro',
                                         random_state=42)
        regressor.fit(X_embedded,
                      y_train)
        preds = regressor.predict(X_val_embedded)
        avg_precision = average_precision_score(y_val, preds)
        report = classification_report(y_val, preds, output_dict=True)

        return avg_precision, report, regressor
