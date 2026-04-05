# evaluate.py

from sklearn.metrics import classification_report, confusion_matrix
import joblib
from preprocessing import load_and_preprocess

X, y = load_and_preprocess('../data/telecom_churn.csv')

model = joblib.load('../model/churn_model.pkl')

pred = model.predict(X)

print(confusion_matrix(y, pred))
print(classification_report(y, pred))