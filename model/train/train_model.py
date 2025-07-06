import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

df = pd.read_csv("data/transactions.csv", sep=',')
df['dataHora'] = pd.to_datetime(df['dataHora'])
df['hora'] = df['dataHora'].dt.hour
df['minuto'] = df['dataHora'].dt.minute
df['dia_da_semana'] = df['dataHora'].dt.dayofweek

y = df['fraude']
X = df.drop(columns=['fraude', 'dataHora'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/fraud_model.pkl")

print("Model trained and saved successfully at: model/fraud_model.pkl")