import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE


def embedded_method(X, y, feature_names):
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    selector = SelectFromModel(model, prefit=True, max_features=30)
    selected_features = selector.get_support()
    return np.array(feature_names)[selected_features]


def filter_method(X, y, feature_names):
    selector = SelectKBest(score_func=chi2, k=30)
    selector.fit(X, y)
    selected_features = selector.get_support()
    return np.array(feature_names)[selected_features]


def wrapper_method(X, y, feature_names):
    model = LogisticRegression(max_iter=100, random_state=42)
    rfe = RFE(model, n_features_to_select=30)
    rfe.fit(X, y)
    selected_features = rfe.get_support()
    return np.array(feature_names)[selected_features]
