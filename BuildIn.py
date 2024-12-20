import numpy as np
from sklearn.tree import DecisionTreeClassifier


def DFS(node_depth, children_left, children_right, is_leaves):
    dfs = [(0, 0)]
    while len(dfs) > 0:
        node_id, depth = dfs.pop()
        node_depth[node_id] = depth
        is_split_node = children_left[node_id] != children_right[node_id]
        if is_split_node:
            dfs.append((children_left[node_id], depth + 1))
            dfs.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True
    return node_depth, is_leaves

def built_in_mine(X, y, feature_names, k=30):
    model = DecisionTreeClassifier()
    model.fit(X, y)

    n_nodes = model.tree_.node_count
    children_left = model.tree_.children_left
    children_right = model.tree_.children_right
    feature = model.tree_.feature

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)

    node_depth, is_leaves = DFS(node_depth, children_left, children_right, is_leaves)
    features_depth = {}
    for i in range(n_nodes):
        if not is_leaves[i]:
            features_depth[feature_names[feature[i]]] = node_depth[i]

    return np.array([item[0] for item in sorted(features_depth.items(), key=lambda item: item[1])[:k]])
