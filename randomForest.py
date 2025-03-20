import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score



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


#define features X and target y

X_rf = encoded_df.drop('considered_successful',axis=1)
y_rf = encoded_df['considered_successful']


X_train_rf , X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

rf_classifier.fit(X_train_rf, y_train_rf)


y_pred_rf = rf_classifier.predict(X_test_rf)

accuracy_rf = accuracy_score(y_test_rf, y_pred_rf)
precision_rf = precision_score(y_test_rf, y_pred_rf, average='weighted')
recall_rf = recall_score(y_test_rf, y_pred_rf, average='weighted')
f1_rf = f1_score(y_test_rf, y_pred_rf, average='weighted')

# Display performance metrics
print(f"Accuracy: {accuracy_rf}")
print(f"Precision: {precision_rf}")
print(f"Recall: {recall_rf}")
print(f"F1-Score: {f1_rf}")