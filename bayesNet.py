import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score




data_path = 'C:\Users\baske\Documents\CSCI_356\mini_project_one_dataset(1).csv'

df = pd.read_csv(data_path)

encoded_df = df.copy()
bool_columns = encoded_df.select_dtypes(include=['bool']).columns
encoded_df[bool_columns] = encoded_df[bool_columns].astype(int)


categorical_columns = encoded_df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    encoded_df[col] = LabelEncoder().fit_transform(encoded_df[col])


features = [
    'has_high_income', 'has_college_degree', 'industry', 'max_degree', 
    'age_group', 'risk_tolerance', 'owns_home', 'has_multiple_income_sources'
]


X = encoded_df[features]
y = encoded_df['considered_successful']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Function to calculate conditional probability
def conditional_prob(data, target, condition, value):
    condition_met = data[data[condition] == value]
    prob = len(condition_met[condition_met[target] == 1]) / len(condition_met) if len(condition_met) > 0 else 0
    return prob


probabilities = {}

probabilities['P(successful)'] = y_train.mean()

# Calculate conditional probabilities
for feature in features:
    probabilities[feature] = {}
    for value in X_train[feature].unique():
        probabilities[feature][value] = conditional_prob(encoded_df, 'considered_successful', feature, value)


def predict(row):
    prob_success = probabilities['P(successful)']
    prob_failure = 1 - prob_success

    for feature in features:
        value = row[feature]
        prob_success *= probabilities[feature].get(value, 1e-6)  # Smoothing for unseen values
        prob_failure *= (1 - probabilities[feature].get(value, 1e-6))

    return 1 if prob_success > prob_failure else 0



y_pred = X_test.apply(predict, axis=1)



# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Display results
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")