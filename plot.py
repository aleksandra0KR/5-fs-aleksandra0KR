import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from BuildIn import built_in_mine

data= open('SMS.tsv', encoding='UTF-8')
file_lines = data.readlines()
labels, raw_inputs = list(zip(*[tuple(file_line.split(maxsplit=1)) for file_line in file_lines]))

count_vectorizer = CountVectorizer(stop_words='english')
count_vector = count_vectorizer.fit_transform(list(raw_inputs))
count_df = pd.DataFrame(count_vector.toarray(), columns=count_vectorizer.get_feature_names_out())

label_mapper = {
    'spam': 0,
    'ham': 1
}
y = np.array([label_mapper[label] for label in labels])

X = count_df.to_numpy()
feature_names = count_df.columns.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

selected_features = built_in_mine(X, y, feature_names)
print("Selected features using filter method:", selected_features)

selected_indices = [list(feature_names).index(f) for f in selected_features]
X_selected = X[:, selected_indices]

kmeans_before = KMeans(n_clusters=2, random_state=42)
kmeans_after = KMeans(n_clusters=2, random_state=42)

clusters_before = kmeans_before.fit_predict(X)
clusters_after = kmeans_after.fit_predict(X_selected)

pca_before = PCA(n_components=2, random_state=42)
pca_after = PCA(n_components=2, random_state=42)

X_pca_before = pca_before.fit_transform(X)
X_pca_after = pca_after.fit_transform(X_selected)

tsne_before = TSNE(n_components=2, random_state=42, init='random')
tsne_after = TSNE(n_components=2, random_state=42, init='random')

X_tsne_before = tsne_before.fit_transform(X)
X_tsne_after = tsne_after.fit_transform(X_selected)


def plot_data(X, y, clusters, title):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.5, label='Real Classes')
    plt.title('Real Classes')

    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', alpha=0.5, label='Clusters')
    plt.title('KMeans Clusters')

    plt.suptitle(title)
    plt.savefig(f'{title}.png')
    plt.show()


plot_data(X_pca_before, y, clusters_before, 'PCA Before Feature Selection')
plot_data(X_pca_after, y, clusters_after, 'PCA After Feature Selection')

plot_data(X_tsne_before, y, clusters_before, 't-SNE Before Feature Selection')
plot_data(X_tsne_after, y, clusters_after, 't-SNE After Feature Selection')
