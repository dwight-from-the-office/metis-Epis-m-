import numpy as np
import pandas  as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder



data_path = 'C:\Users\baske\Documents\CSCI_356\mini_project_one_dataset(1).csv'

df = pd.read_csv(data_path)

encoded_df = df.copy()

bool_columns = encoded_df.select_dtypes(include=['bool']).columns
encoded_df[bool_columns] = encoded_df[bool_columns].astype(int)


categorical_columns = encoded_df.select_dtypes(include=['object']).columns
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    encoded_df[col] = le.fit_transform(encoded_df[col])
    label_encoders[col] = le


X = encoded_df.drop("considered_successful", axis=1)
y = encoded_df['considered_successful']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb_classifier = GaussianNB()


nb_classifier.fit(X_train, y_train)

y_pred = nb_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(accuracy, precision, recall,f1)



