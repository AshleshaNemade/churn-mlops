import optuna
import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# Load data
df = pd.read_csv("data/processed/train.csv")

X = df.drop("Churn", axis=1)
y = df["Churn"]

os.makedirs("models", exist_ok=True)

joblib.dump(X.columns.tolist(), "models/features.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


def objective(trial):

    model_type = trial.suggest_categorical(
        "model", ["logistic", "random_forest", "xgboost"]
    )

    if model_type == "logistic":
        C = trial.suggest_float("C", 0.01, 10.0, log=True)
        model = LogisticRegression(C=C, max_iter=1000)

    elif model_type == "random_forest":
        n_estimators = trial.suggest_int("n_estimators", 50, 200)
        max_depth = trial.suggest_int("max_depth", 3, 15)
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth
        )

    else:  # XGBoost
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)

        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            use_label_encoder=False,
            eval_metric="logloss"
        )

    model.fit(X_train, y_train)

    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)

    return auc


# Run optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

best_params = study.best_params
best_model_type = best_params.pop("model")


# Train final model
if best_model_type == "logistic":
    model = LogisticRegression(**best_params, max_iter=1000)

elif best_model_type == "random_forest":
    model = RandomForestClassifier(**best_params)

else:
    model = XGBClassifier(
        **best_params,
        use_label_encoder=False,
        eval_metric="logloss"
    )

model.fit(X_train, y_train)

# Evaluate
preds = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, preds)


# Log to MLflow
with mlflow.start_run(run_name="optuna_all_models"):
    mlflow.log_param("model_type", best_model_type)
    mlflow.log_params(best_params)
    mlflow.log_metric("roc_auc", auc)

    mlflow.sklearn.log_model(model, "model")


# Save model + features
joblib.dump(model, "models/best_model.pkl")
joblib.dump(X.columns.tolist(), "models/features.pkl")

print("Best model:", best_model_type)
print("Best params:", best_params)
print("Best ROC-AUC:", auc)