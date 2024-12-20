import numpy as np
from scipy.stats import pearsonr

def filter_mine(X, y, features, k=30):
    scores = []
    for f in range(X.shape[1]):
        feature_column = X[:, f]
        if np.all(feature_column != feature_column[0]):
            corr, _ = pearsonr(feature_column, y)
            scores.append(abs(corr))
        else:
            scores.append(0)

    sorted_features_filter = np.argsort(scores)[::-1]
    top_features = np.array(features)[sorted_features_filter[:k]]
    return top_features
