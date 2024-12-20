import numpy as np
from sklearn.tree import DecisionTreeClassifier

def wrapper_mine(X, y, feature_names, k=30):
    features = set()
    for _ in range(k):
        feature_scores = []
        for i in range(X.shape[1]):
            if i not in features:
                model = DecisionTreeClassifier(max_depth=10, min_samples_split=2)
                considered_set = list(features.union({i}))
                model.fit(X[:, considered_set], y)
                feature_scores.append(model.score(X[:, considered_set], y))
            else:
                feature_scores.append(-10000000)
        features.add(np.argmax(feature_scores))
    return feature_names[list(features)]
