import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

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

models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=100, random_state=42),
    'DecisionTree': DecisionTreeClassifier(random_state=42)
}

#for name, model in models.items():
#    model.fit(X_train, y_train)
#    y_pred = model.predict(X_test)
#    print(f"{name} accuracy before feature selection: {accuracy_score(y_test, y_pred)}")

selected_features_filter = built_in_mine(X_train, y_train, feature_names, k=30)
selected_indices_filter = [list(feature_names).index(f) for f in selected_features_filter]
X_train_filter = X_train[:, selected_indices_filter]
X_test_filter = X_test[:, selected_indices_filter]

selected_features_embedded = filter_mine(X_train, y_train, feature_names, k=30)
selected_indices_embedded = [list(feature_names).index(f) for f in selected_features_embedded]
X_train_embedded = X_train[:, selected_indices_embedded]
X_test_embedded = X_test[:, selected_indices_embedded]

selected_features_wrapper = wrapper_mine(X_train, y_train, feature_names, k=30)
selected_indices_wrapper = [list(feature_names).index(f) for f in selected_features_wrapper]
X_train_wrapper = X_train[:, selected_indices_wrapper]
X_test_wrapper = X_test[:, selected_indices_wrapper]

for name, model in models.items():
    model.fit(X_train_filter, y_train)
    y_pred = model.predict(X_test_filter)
    print(f"{name} accuracy after filter method: {accuracy_score(y_test, y_pred)}")

    model.fit(X_train_embedded, y_train)
    y_pred = model.predict(X_test_embedded)
    print(f"{name} accuracy after embedded method: {accuracy_score(y_test, y_pred)}")

    model.fit(X_train_wrapper, y_train)
    y_pred = model.predict(X_test_wrapper)
    print(f"{name} accuracy after wrapper method: {accuracy_score(y_test, y_pred)}")
