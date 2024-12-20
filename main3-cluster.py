import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import train_test_split
from BuildIn import built_in_mine
from Filter import filter_mine
from Wrapper import wrapper_mine

data = open('SMS.tsv', encoding='UTF-8')
file_lines = data.readlines()
labels, raw_inputs = list(zip(*[tuple(file_line.split(maxsplit=1)) for file_line in file_lines]))

vector = CountVectorizer(stop_words='english')
count_vector = vector.fit_transform(list(raw_inputs))
vector_df = pd.DataFrame(count_vector.toarray(), columns=vector.get_feature_names_out())

label_mapper = {
    'spam': 0,
    'ham': 1
}
y = np.array([label_mapper[label] for label in labels])
X = vector_df.to_numpy()
feature_names = vector_df.columns.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

kmeans_before = KMeans(n_clusters=2, random_state=42)
clusters_before = kmeans_before.fit_predict(X_train)

ari_no_selection = adjusted_rand_score(y_train, clusters_before)
silhouette_no_selection = silhouette_score(X_train, clusters_before)
print(f"Adjusted Rand Index before feature selection: {ari_no_selection}")
print(f"Silhouette Score before feature selection: {silhouette_no_selection}")
'''
Adjusted Rand Index before feature selection: -0.054515494142974213
Silhouette Score before feature selection: 0.17299745225834856
'''


selected_features = built_in_mine(X, y, feature_names)
print("Selected features using filter method:", selected_features)

selected_indices = [list(feature_names).index(f) for f in selected_features]
X_train_selected = X_train[:, selected_indices]
X_test_selected = X_test[:, selected_indices]

kmeans_after = KMeans(n_clusters=2, random_state=42)
clusters_after = kmeans_after.fit_predict(X_train_selected)


ari_with_selection = adjusted_rand_score(y_train, clusters_after)
silhouette_with_selection = silhouette_score(X_train_selected, clusters_after)
print(f"Adjusted Rand Index after feature selection: {ari_with_selection}")
print(f"Silhouette Score after feature selection: {silhouette_with_selection}")

'''
Adjusted Rand Index after feature selection: 0.2501740036737811
Silhouette Score after feature selection: 0.7634013246436302
'''


selected_features = filter_mine(X, y, feature_names)
print("Selected features using filter method:", selected_features)


selected_indices = [list(feature_names).index(f) for f in selected_features]
X_train_selected = X_train[:, selected_indices]
X_test_selected = X_test[:, selected_indices]


kmeans_after = KMeans(n_clusters=2, random_state=42)
clusters_after = kmeans_after.fit_predict(X_train_selected)


ari_with_selection = adjusted_rand_score(y_train, clusters_after)
silhouette_with_selection = silhouette_score(X_train_selected, clusters_after)
print(f"Adjusted Rand Index after feature selection: {ari_with_selection}")
print(f"Silhouette Score after feature selection: {silhouette_with_selection}")


selected_features = wrapper_mine(X, y, feature_names)
print("Selected features using filter method:", selected_features)

selected_indices = [list(feature_names).index(f) for f in selected_features]
X_train_selected = X_train[:, selected_indices]
X_test_selected = X_test[:, selected_indices]


kmeans_after = KMeans(n_clusters=2, random_state=42)
clusters_after = kmeans_after.fit_predict(X_train_selected)


ari_with_selection = adjusted_rand_score(y_train, clusters_after)
silhouette_with_selection = silhouette_score(X_train_selected, clusters_after)
print(f"Adjusted Rand Index after feature selection: {ari_with_selection}")
print(f"Silhouette Score after feature selection: {silhouette_with_selection}")


'''
Adjusted Rand Index after feature selection: 0.249688301230111
Silhouette Score after feature selection: 0.7731011876543235
'''
