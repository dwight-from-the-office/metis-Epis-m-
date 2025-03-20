import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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



X_knn = encoded_df.drop('considered_successful', axis=1)
y_knn = encoded_df['considered_successful']


X_train_knn , X_test_knn, y_train_knn, y_test_knn = train_test_split(X_knn, y_knn, test_size=0.2, random_state=42)

knn_classifier = KNeighborsClassifier(n_neighbors=5)

knn_classifier.fit(X_train_knn, y_train_knn)

y_pred_knn = knn_classifier.predict(X_test_knn)

accuracy_knn = accuracy_score(y_test_knn, y_pred_knn)
precision_knn = precision_score(y_test_knn, y_pred_knn, average='weighted')
recall_knn = recall_score(y_test_knn, y_pred_knn, average='weighted')
f1_knn = f1_score(y_test_knn, y_pred_knn, average='weighted')

# Display performance metrics
print(f"Accuracy: {accuracy_knn}")
print(f"Precision: {precision_knn}")
print(f"Recall: {recall_knn}")
print(f"F1-Score: {f1_knn}")