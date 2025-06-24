import streamlit as st
import numpy as np
import pickle
import os
from train import train_and_save_model
import mlflow
import pandas as pd
import streamlit.components.v1 as components

st.set_page_config(page_title="MLOps 대시보드", layout="wide")

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

page = st.sidebar.selectbox("📚 페이지 선택", ["🔍 예측하기", "⚙️ MLOps 대시보드"])

if page == "🔍 예측하기":
    st.title("🔍 Iris 예측기")
    model = load_model()

    sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.1)
    sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.5)
    petal_length = st.slider("Petal Length", 1.0, 7.0, 1.4)
    petal_width = st.slider("Petal Width", 0.1, 2.5, 0.2)

    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    if st.button("📊 예측하기"):
        prediction = model.predict(input_data)
        class_names = ["Setosa", "Versicolor", "Virginica"]
        st.success(f"예측 결과: **{class_names[prediction[0]]}**")

elif page == "⚙️ MLOps 대시보드":
    st.title("⚙️ MLOps 대시보드")

    if st.button("🚀 모델 재학습"):
        acc, params = train_and_save_model()
        st.success(f"모델 재학습 완료. 정확도: {acc:.3f}")
        st.json(params)

    if os.path.exists("model.pkl"):
        st.info("✅ model.pkl 파일 존재")
    else:
        st.warning("❗ model.pkl 파일이 없습니다. 먼저 학습하세요.")

    st.subheader("📈 MLflow 실험 요약")
    try:
        mlflow.set_tracking_uri("http://mlflow:5000")
        experiment = mlflow.get_experiment_by_name("Iris_Classification")
        if experiment:
            runs = mlflow.search_runs(experiment.experiment_id)
            st.dataframe(runs[["run_id", "params.n_estimators", "params.max_depth", "metrics.accuracy", "start_time"]])
        else:
            st.warning("MLflow에 기록된 실험이 없습니다.")
    except Exception as e:
        st.error(f"MLflow 연동 실패: {e}")

    st.subheader("🔍 MLflow UI 내장 보기")
    components.iframe("http://localhost:5000", height=800, scrolling=True)