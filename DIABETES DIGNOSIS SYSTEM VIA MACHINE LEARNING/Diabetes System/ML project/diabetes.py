import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('diabetes.csv')

X = data.drop('Outcome', axis=1) 
y = data['Outcome'] 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))


import joblib
joblib.dump(model, 'diabetes_model.pkl')


def predict_diabetes(input_data):
    input_data = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_data)
    return 'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'

# Example Prediction
example = [1,85,66,29,0,26.6,0.351,31]
result = predict_diabetes(example)
print(f'Prediction: {result}')
