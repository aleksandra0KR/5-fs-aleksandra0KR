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
kmeans_before.fit(X_train)
clusters_before_train = kmeans_before.predict(X_train)
clusters_before_test = kmeans_before.predict(X_test)

external_estimation_before_selection = adjusted_rand_score(y_test, clusters_before_test)
internal_estimation_before_selection = silhouette_score(X_train, clusters_before_train)
print(f"Adjusted Rand Index: {external_estimation_before_selection}")
print(f"Silhouette Score: {internal_estimation_before_selection}")
'''
Adjusted Rand Index: -0.054515494142974213
Silhouette Score: 0.17299745225834856
'''

selected_features_built = built_in_mine(X, y, feature_names)
selected_indices = [list(feature_names).index(f) for f in selected_features_built]
X_train_selected = X_train[:, selected_indices]
X_test_selected = X_test[:, selected_indices]

kmeans_after = KMeans(n_clusters=2, random_state=42)
kmeans_after.fit(X_train_selected)
clusters_after_train = kmeans_after.predict(X_train_selected)
clusters_after_test = kmeans_after.predict(X_test_selected)

external_estimation_after_selection = adjusted_rand_score(y_test, clusters_after_test)
internal_estimation_after_selection = silhouette_score(X_train_selected, clusters_after_train)
print(f"Adjusted Rand Index: {external_estimation_after_selection}")
print(f"Silhouette Score: {internal_estimation_after_selection}")
'''
Adjusted Rand Index: 0.2501740036737811
Silhouette Score: 0.7634013246436302
'''

selected_features_filter = filter_mine(X, y, feature_names)
selected_indices = [list(feature_names).index(f) for f in selected_features_filter]
X_train_selected = X_train[:, selected_indices]
X_test_selected = X_test[:, selected_indices]

kmeans_after = KMeans(n_clusters=2, random_state=42)
kmeans_after.fit(X_train_selected)
clusters_after_train = kmeans_after.predict(X_train_selected)
clusters_after_test = kmeans_after.predict(X_test_selected)

external_estimation_after_selection = adjusted_rand_score(y_test, clusters_after_test)
internal_estimation_after_selection = silhouette_score(X_train_selected, clusters_after_train)
print(f"Adjusted Rand Index: {external_estimation_after_selection}")
print(f"Silhouette Score: {internal_estimation_after_selection}")
'''
Adjusted Rand Index: 0.259674995429451
Silhouette Score: 0.7911033743614592
'''


selected_features_wrap = wrapper_mine(X, y, feature_names)
selected_indices = [list(feature_names).index(f) for f in selected_features_wrap]
X_train_selected = X_train[:, selected_indices]
X_test_selected = X_test[:, selected_indices]

kmeans_after = KMeans(n_clusters=2, random_state=42)
kmeans_after.fit(X_train_selected)
clusters_after_train = kmeans_after.predict(X_train_selected)
clusters_after_test = kmeans_after.predict(X_test_selected)

external_estimation_after_selection = adjusted_rand_score(y_test, clusters_after_test)
internal_estimation_after_selection = silhouette_score(X_train_selected, clusters_after_train)
print(f"Adjusted Rand: {external_estimation_after_selection}")
print(f"Silhouette Score: {internal_estimation_after_selection}")
'''
Adjusted Rand: 0.249688301230111
Silhouette Score: 0.7731011876543235
'''
