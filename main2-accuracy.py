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

X_train, X_test, y_train, y_test = train_test_split(vector_df, y, test_size=0.2, random_state=42)

models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=100, random_state=42),
    'DecisionTree': DecisionTreeClassifier(random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name} accuracy before feature selection: {accuracy_score(y_test, y_pred)}")

selected_features_filter = built_in_mine(X_train, y_train, X_train.columns, k=10)
X_train_filter = X_train[selected_features_filter]
X_test_filter = X_test[selected_features_filter]

selected_features_embedded = filter_mine(X_train, y_train, X_train.columns, k=10)
X_train_embedded = X_train[selected_features_embedded]
X_test_embedded = X_test[selected_features_embedded]

selected_features_wrapper = wrapper_mine(X_train, y_train, X_train.columns, k=10)
X_train_wrapper = X_train[selected_features_wrapper]
X_test_wrapper = X_test[selected_features_wrapper]

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

'''
RandomForest accuracy before feature selection: 0.9757847533632287
LogisticRegression accuracy before feature selection: 0.9820627802690582
DecisionTree accuracy before feature selection: 0.9721973094170404
RandomForest accuracy after filter method: 0.9426008968609866
RandomForest accuracy after embedded method: 0.9434977578475336
LogisticRegression accuracy after filter method: 0.9426008968609866
LogisticRegression accuracy after embedded method: 0.9408071748878923
DecisionTree accuracy after filter method: 0.9426008968609866
DecisionTree accuracy after embedded method: 0.9434977578475336
'''
