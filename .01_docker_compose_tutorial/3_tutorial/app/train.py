import pickle
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_and_save_model():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    # MLflow 서버 URI 설정
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("Iris_Classification")

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, max_depth=3)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 3)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, artifact_path="model")

        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)

    return acc, model.get_params()