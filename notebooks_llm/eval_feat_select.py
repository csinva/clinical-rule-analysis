from sklearn.model_selection import train_test_split

# import sklearn tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from imodels import HSTreeClassifierCV
import numpy as np
from sklearn.model_selection import cross_val_score
import warnings


def evaluate_features(X, y, seed=42, test_size=0.3, class_weight=500):
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=test_size, random_state=seed, stratify=y
    # )
    # print(y_train.mean(), y_test.mean(), y_test.sum())
    m = DecisionTreeClassifier(random_state=seed, class_weight={0: 1, 1: class_weight})

    # compute cross validation scores
    met_scores = {}
    for scoring in ["roc_auc", "accuracy", "f1", "precision", "recall"]:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Precision is ill-defined*")
            met_scores[scoring] = np.mean(
                cross_val_score(m, X, y, cv=5, scoring=scoring)
            )
        # print(f"{scoring}: {scores.mean():.3f} +/- {scores.std():.3f}")

    # m.fit(X_train, y_train)

    # y_pred = m.predict(X_test)
    # y_pred_proba = m.predict_proba(X_test)[:, 1]

    # # calculate metrics
    # met_scores = {
    #     "accuracy": accuracy_score(y_test, y_pred),
    #     "f1": f1_score(y_test, y_pred),
    #     "precision": precision_score(y_test, y_pred),
    #     "recall": recall_score(y_test, y_pred),
    #     "roc_auc": roc_auc_score(y_test, y_pred_proba),
    # }
    return met_scores
