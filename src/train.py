import mlflow
import mlflow.sklearn
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from preprocessing import preprocess
from models import get_models
from evaluate import evaluate_model

df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = preprocess(df)

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = get_models()

best_model = None
best_score = 0
best_model_name = ""

for name, model in models.items():

    with mlflow.start_run(run_name=name):

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = evaluate_model(y_test, y_pred, y_prob)

        mlflow.log_param("model", name)

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        mlflow.sklearn.log_model(model, name)

        print(f"{name} ROC-AUC: {metrics['roc_auc']}")

        if metrics["roc_auc"] > best_score:
            best_score = metrics["roc_auc"]
            best_model = model
            best_model_name = name

joblib.dump(best_model, "models/best_model.pkl")

print(f"\nBest Model: {best_model_name} | ROC-AUC: {best_score}")